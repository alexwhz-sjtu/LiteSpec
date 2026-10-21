"""Final-output-only CE/L1 training for RealSemi."""

import torch

from ._dflash_common import DFlashTrainingBase
from .attention_masks import create_dflash_flex_mask, create_dflash_sdpa_mask
from .block_metrics import fsdp_global_weighted_mean
from .chunking import checkpointed_chunk_reduce
from .losses import weighted_ce_l1
from .outputs import RatioTerms, TrainingOutput


class OnlineRealSemiModel(DFlashTrainingBase):
    def __init__(self, *args, ce_loss_alpha=0.1, l1_loss_alpha=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        if (
            min(self.ce_loss_alpha, self.l1_loss_alpha) < 0
            or self.ce_loss_alpha + self.l1_loss_alpha <= 0
        ):
            raise ValueError(
                "RealSemi loss coefficients must be nonnegative with a positive sum"
            )
        self.lm_head.requires_grad_(False)
        self.embed_tokens.requires_grad_(False)

    def forward(
        self,
        input_ids,
        hidden_states,
        loss_mask,
        target_last_hidden_states=None,
        max_valid_anchors=None,
        collect_detailed_metrics=True,
    ):
        if target_last_hidden_states is None:
            raise ValueError("RealSemi CE/L1 requires frozen target last hidden states")
        batch, seq_len = input_ids.shape
        anchors, keep = self._sample_anchor_positions(
            seq_len, loss_mask, input_ids.device, max_valid_anchors=max_valid_anchors
        )
        if anchors.size(1) == 0:
            raise ValueError("No valid RealSemi training anchors")
        offsets = torch.arange(1, self.block_size, device=input_ids.device)
        indices = anchors[..., None] + offsets
        safe = indices.clamp(max=seq_len - 1)
        expanded_ids = input_ids[:, None].expand(-1, anchors.size(1), -1)
        labels = expanded_ids.gather(2, safe)
        predecessors = expanded_ids.gather(2, (indices - 1).clamp(max=seq_len - 1))
        weights = (keep[..., None] & (indices < seq_len)).float()
        weights = (
            weights * loss_mask[:, None].expand_as(expanded_ids).gather(2, safe).float()
        )
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            weights = weights * torch.exp(
                -(offsets.float() - 1) / self.loss_decay_gamma
            )
        noise = self._create_noise_embed(input_ids, anchors, keep)
        context_positions = torch.arange(seq_len, device=input_ids.device)[None].expand(
            batch, -1
        )
        positions = torch.cat(
            (context_positions, self._create_position_ids(anchors)), -1
        )
        mask_builder = (
            create_dflash_flex_mask
            if self.attention_backend == "flex_attention"
            else create_dflash_sdpa_mask
        )
        mask_args = dict(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=seq_len,
            block_size=self.block_size,
            device=input_ids.device,
        )
        if self.attention_backend == "flex_attention":
            mask_args["flex_backend"] = self.flex_attention_backend
        kwargs = (
            {"kernel_options": self.flex_kernel_options}
            if self.flex_kernel_options
            else {}
        )
        predictions = self.draft_model(
            noise_embedding=noise,
            target_hidden=hidden_states,
            position_ids=positions,
            attention_mask=mask_builder(**mask_args),
            previous_token_embeddings=self.embed_tokens(predecessors),
            anchor_positions=anchors,
            block_keep_mask=keep,
            flex_backend=self.flex_attention_backend,
            **kwargs,
        )
        # Teacher k-1 is supervision only; it is never passed to the draft input.
        teacher_hidden = self._aligned_target_hidden(target_last_hidden_states, safe)
        # Mask invalid queries before their head projection as well as their loss.
        predictions = torch.where(
            weights[..., None] > 0, predictions, torch.zeros_like(predictions)
        )

        def objective(hidden, gold, weight, teacher):
            logits = self.lm_head(hidden).float()
            with torch.no_grad():
                teacher_logits = self.lm_head(teacher).float()
            ce, l1, _, per_l1 = weighted_ce_l1(logits, gold, teacher_logits, weight)
            valid = weight > 0
            if collect_detailed_metrics:
                correct = ((logits.argmax(-1) == gold) & valid).float()
                overlap = (1.0 - 0.5 * per_l1).clamp(0, 1)
                accuracy = correct.sum()
                overlap_sum = (overlap * weight).sum()
                prefix = correct.cumprod(-1).sum()
                chain = (overlap * valid).cumprod(-1).sum()
            else:
                accuracy = overlap_sum = prefix = chain = ce.detach().new_zeros(())
            return (
                ce,
                l1,
                accuracy.detach(),
                overlap_sum.detach(),
                prefix.detach(),
                chain.detach(),
            )

        ce, l1, accuracy, overlap, prefix, chain = checkpointed_chunk_reduce(
            objective,
            predictions,
            labels,
            weights,
            teacher_hidden,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        numerator = self.ce_loss_alpha * ce + self.l1_loss_alpha * l1
        denominator = weights.sum().detach()
        metrics = {
            "objective/ce": RatioTerms(ce.detach(), denominator),
            "objective/l1": RatioTerms(l1.detach(), denominator),
            "objective/tv": RatioTerms(0.5 * l1.detach(), denominator),
        }
        if collect_detailed_metrics:
            metrics.update(
                {
                    # Public metrics use DFlash's anchor-inclusive convention.
                    # Keep the existing suffix-only TF diagnostics for continuity.
                    "hard_label/block_accepted_length": RatioTerms(
                        prefix + keep.sum().detach(), keep.sum().detach()
                    ),
                    "expected_accept_len": RatioTerms(
                        chain + keep.sum().detach(), keep.sum().detach()
                    ),
                    "realsemi/tf/token_accuracy": RatioTerms(
                        accuracy, (weights > 0).sum().detach()
                    ),
                    "realsemi/tf/distribution_overlap": RatioTerms(
                        overlap, denominator
                    ),
                    "realsemi/tf/block_accepted_length": RatioTerms(
                        prefix, keep.sum().detach()
                    ),
                    "realsemi/tf/overlap_chain_length": RatioTerms(
                        chain, keep.sum().detach()
                    ),
                }
            )
        return TrainingOutput(
            loss=fsdp_global_weighted_mean(numerator, denominator, self.process_group),
            loss_terms=RatioTerms(numerator, denominator),
            ratio_metrics=metrics,
        )
