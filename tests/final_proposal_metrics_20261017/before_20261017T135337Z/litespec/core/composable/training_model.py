"""One FSDP forward owns all trainable fusion, backbone, heads and losses."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.modeling.draft.composable.contracts import (
    BackboneOutput,
    ModelInputs,
    SharedOps,
)

from .batch import prepare_layout
from .metrics import acceptance_metric_terms
from .objectives import build_objective
from .objectives.base import Supervision
from .objectives.selector import build_selector_objective


class ComposableTrainingModel(nn.Module):
    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id,
        num_anchors=32,
        objective_chunk_blocks=32,
        **kwargs,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.mask_token_id = int(mask_token_id)
        self.num_anchors = int(num_anchors)
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        recipe = draft_model.config.composable_config
        self.objective = build_objective(
            recipe["objective"], recipe["proposal"]["type"]
        )
        self.selector_enabled = recipe["proposal"]["type"] == "selector"
        self.selector_objective = (
            build_selector_objective(recipe) if self.selector_enabled else None
        )

    def forward(
        self,
        input_ids,
        target_hidden_states,
        loss_mask,
        target_last_hidden_states=None,
        layout=None,
        window_denominators=None,
        epoch_progress=0.0,
        selector_loss_alpha=1.0,
        collect_detailed_metrics=False,
        context_valid_mask=None,
    ):
        model = self.draft_model
        if collect_detailed_metrics and target_last_hidden_states is None:
            raise ValueError(
                "Acceptance metrics require target_last_hidden_states"
            )
        if layout is None:
            layout = prepare_layout(
                input_ids,
                loss_mask,
                model.block_size,
                self.num_anchors,
                complete_blocks=self.objective.recipe["type"] == "bv",
            )
        batch_size, num_anchors = layout.anchor_positions.shape
        noise_ids = input_ids.new_full(
            (batch_size, num_anchors, model.block_size), self.mask_token_id
        )
        noise_ids[..., 0] = input_ids.gather(1, layout.anchor_positions).masked_fill(
            ~layout.block_keep_mask, self.mask_token_id
        )
        inputs = ModelInputs(
            target_hidden_states=target_hidden_states,
            draft_embeddings=self.embed_tokens(noise_ids.flatten(1)),
            anchor_positions=layout.anchor_positions,
            block_keep_mask=layout.block_keep_mask,
            mode="features",
            context_valid_mask=context_valid_mask,
        )
        prediction = model(
            inputs, shared_ops=SharedOps(self.embed_tokens, self.lm_head)
        )
        # Chunk by WHOLE blocks. NLA's per-sample count spans all chunks.
        chunk_size = self.objective_chunk_blocks or num_anchors
        main_loss_numerator = prediction.draft_hidden_states.sum() * 0.0
        main_loss_denominator = main_loss_numerator.detach().clone()
        selector_loss_numerator = main_loss_numerator
        selector_loss_denominator = main_loss_denominator.clone()
        correct_num, accuracy_den = (
            main_loss_denominator.clone(),
            main_loss_denominator.clone(),
        )
        mathematical_num = main_loss_denominator.clone()
        # Keep block counts exact when the model uses BF16 (odd counts >256
        # cannot be represented by a BF16 scalar).
        walk_num = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        expected_num = walk_num.clone()
        block_den = walk_num.clone()
        for start in range(0, num_anchors, chunk_size):
            end = min(start + chunk_size, num_anchors)
            block_hidden = prediction.backbone_hidden_states.reshape(
                batch_size, num_anchors, model.block_size, -1
            )[:, start:end]
            labels = layout.labels[:, start:end]
            previous_token_ids = layout.previous_token_ids[:, start:end]
            mask = layout.prediction_mask[:, start:end]
            teacher_hidden = (
                None
                if target_last_hidden_states is None
                or not (self.objective.needs_teacher or collect_detailed_metrics)
                else torch.gather(
                    target_last_hidden_states,
                    1,
                    (layout.label_indices[:, start:end] - 1)
                    .clamp_min(0)
                    .flatten(1)[..., None]
                    .expand(-1, -1, model.config.hidden_size),
                ).reshape(*labels.shape, -1)
            )

            # Bind layout slices in a callable factory: checkpoint replay must not
            # capture loop variables that will change before backward.
            def make_chunk(labels, previous_token_ids, mask, teacher_hidden):
                def run_chunk(block_hidden):
                    chunk_inputs = replace(
                        inputs,
                        anchor_positions=layout.anchor_positions[
                            :, : block_hidden.size(1)
                        ],
                        block_keep_mask=layout.block_keep_mask[
                            :, : block_hidden.size(1)
                        ],
                        previous_token_ids=previous_token_ids,
                        mode="teacher_forcing",
                    )
                    features = BackboneOutput(block_hidden.flatten(1, 2))
                    context = SimpleNamespace(inputs=chunk_inputs)
                    output = model.proposal(
                        features, context, SharedOps(self.embed_tokens, self.lm_head)
                    )
                    teacher_logits = None
                    if self.objective.needs_teacher or collect_detailed_metrics:
                        if teacher_hidden is None:
                            raise ValueError(
                                "This objective requires target_last_hidden_states"
                            )
                        with torch.no_grad():
                            teacher_logits = self.lm_head(teacher_hidden)
                    supervision = Supervision(
                        labels,
                        mask,
                        teacher_logits,
                        layout.sample_prediction_counts,
                        epoch_progress,
                    )
                    objective_prediction = (
                        replace(output, draft_logits=output.base_logits)
                        if self.selector_enabled
                        else output
                    )
                    terms = self.objective(objective_prediction, supervision)
                    auxiliary = objective_prediction.draft_logits.sum() * 0
                    auxiliary_den = mask.new_zeros(())
                    if self.selector_objective is not None:
                        selector_terms = self.selector_objective(
                            output, supervision
                        ).loss_terms
                        auxiliary, auxiliary_den = (
                            selector_terms.numerator,
                            selector_terms.denominator,
                        )
                    with torch.no_grad():
                        correct = (
                            (objective_prediction.draft_logits.argmax(-1) == labels)
                            * mask
                        ).sum()
                        metric = terms.ratio_metrics.get("bv/mathematical_loss")
                        mathematical = (
                            metric.numerator
                            if metric is not None
                            else mask.new_zeros(())
                        )
                    if collect_detailed_metrics:
                        walk, expected, blocks = acceptance_metric_terms(
                            model.proposal,
                            output,
                            teacher_logits,
                            labels,
                            previous_token_ids,
                            mask,
                        )
                    else:
                        walk = expected = blocks = mask.new_zeros(())
                    return (
                        terms.loss_terms.numerator,
                        terms.loss_terms.denominator,
                        auxiliary,
                        auxiliary_den,
                        correct,
                        mask.sum(),
                        mathematical,
                        walk,
                        expected,
                        blocks,
                    )

                return run_chunk

            operation = make_chunk(labels, previous_token_ids, mask, teacher_hidden)
            values = (
                checkpoint(operation, block_hidden, use_reentrant=False)
                if self.training and torch.is_grad_enabled()
                else operation(block_hidden)
            )
            main_loss_numerator = main_loss_numerator + values[0]
            if not self.objective.sample_reduction or start == 0:
                main_loss_denominator = main_loss_denominator + values[1]
            selector_loss_numerator, selector_loss_denominator = (
                selector_loss_numerator + values[2],
                selector_loss_denominator + values[3],
            )
            correct_num, accuracy_den = (
                correct_num + values[4],
                accuracy_den + values[5],
            )
            mathematical_num = mathematical_num + values[6]
            walk_num = walk_num + values[7]
            expected_num = expected_num + values[8]
            block_den = block_den + values[9]
        if self.selector_enabled:
            if window_denominators is None:
                # Direct, single-forward API; distributed runner always supplies window totals.
                reference = main_loss_numerator.new_tensor(batch_size)
                denominators = torch.stack(
                    (reference, main_loss_denominator, selector_loss_denominator)
                )
            else:
                denominators = window_denominators.to(main_loss_numerator)
            reference, main_global, selector_global = denominators.unbind()
            numerator = reference * (
                main_loss_numerator / main_global.clamp_min(1e-12)
                + selector_loss_alpha
                * selector_loss_numerator
                / selector_global.clamp_min(1e-12)
            )
            denominator = main_loss_numerator.new_tensor(batch_size)
        else:
            numerator, denominator = main_loss_numerator, main_loss_denominator
        metrics = {
            "objective/main": RatioTerms(
                main_loss_numerator.detach(), main_loss_denominator.detach()
            ),
            "hard_label/token_accuracy": RatioTerms(
                correct_num.detach(), accuracy_den.detach()
            ),
        }
        if collect_detailed_metrics:
            metrics["hard_label/block_accepted_length"] = RatioTerms(
                walk_num.detach(), block_den.detach()
            )
            metrics["expected_accept_len"] = RatioTerms(
                expected_num.detach(), block_den.detach()
            )
        if self.selector_enabled:
            metrics["objective/selector_ce"] = RatioTerms(
                selector_loss_numerator.detach(), selector_loss_denominator.detach()
            )
        if self.objective.recipe["type"] == "bv":
            metrics["bv/mathematical_loss"] = RatioTerms(
                mathematical_num.detach(), main_loss_denominator.detach()
            )
        return TrainingOutput(
            numerator / denominator.clamp_min(1e-12),
            RatioTerms(numerator, denominator),
            metrics,
        )
