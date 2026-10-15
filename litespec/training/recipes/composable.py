"""Composable adapter: runtime/data only; model and loss math live in packages."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist

from litespec.core.composable.batch import (
    PreparedLogicalBatch,
    cpu_copy,
    prepare_layout,
    tensor_bytes,
)
from litespec.core.composable.objectives import build_objective
from litespec.core.composable.objectives.selector import build_selector_objective
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.modeling.config_utils import load_text_model_config
from litespec.modeling.draft.composable import (
    ComposableDraftConfig,
    ComposableDraftModel,
)
from litespec.training.config import add_common_args, validate_common_args
from litespec.training.logging import get_tracker_class
from litespec.training.recipes._dflash_common import (
    DFlashFamilyTrainingAdapter,
    prepare_dflash_family_features,
    selector_alpha,
)


def parse_args(mode=None, argv=None):
    parser = argparse.ArgumentParser(description="Train a composable parallel draft")
    add_common_args(parser)
    parser.set_defaults(attention_backend="sdpa", num_anchors=32)
    parser.add_argument("--draft-config", default="configs/composable_draft.json")
    parser.add_argument("--init-from")
    parser.add_argument(
        "--override-attention-backend", choices=["eager", "sdpa", "flex_attention"]
    )
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this many optimizer updates, preserving the full-epoch LR schedule.",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--objective-chunk-blocks", type=int, default=32)
    parser.add_argument("--window-buffer-mb", type=int, default=256)
    parser.add_argument("--selector-loss-alpha", type=float, default=1.0)
    parser.add_argument("--selector-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--selector-ramp-ratio", type=float, default=0.0)
    parser.add_argument("--bv-sampling-provenance", choices=["target_full_vocab_t1"])
    args = parser.parse_args(argv)
    args.algorithm = "composable"
    validate_common_args(parser, args)
    if args.objective_chunk_blocks < 0 or args.window_buffer_mb <= 0:
        parser.error("chunk blocks must be nonnegative and window-buffer-mb positive")
    if (
        args.selector_loss_alpha < 0
        or not 0 <= args.selector_warmup_ratio <= 1
        or not 0 <= args.selector_ramp_ratio <= 1
    ):
        parser.error("Invalid selector weight/schedule")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    args.require_target_last_hidden = True
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


class ComposableTrainingAdapter(DFlashFamilyTrainingAdapter):
    def phases(self, args):
        from dataclasses import replace

        phases = super().phases(args)
        if args.max_steps is not None:
            phases[0] = replace(
                phases[0],
                max_batches=args.max_steps * args.accumulation_steps,
                schedule_full_epochs=True,
            )
        return phases

    def build_config(self, args, *, source_config=None, model_role=None):
        target = (
            load_text_model_config(
                args.target_model_path, trust_remote_code=args.trust_remote_code
            )
            if source_config is None
            else copy.deepcopy(source_config)
        )
        payload = json.loads(Path(args.draft_config).read_text())
        recipe = copy.deepcopy(payload.get("composable_config", payload))
        for layer in recipe["backbone"]["layers"]:
            attention = layer["attention"]
            attention.setdefault("backend", args.attention_backend)
            if args.override_attention_backend is not None:
                attention["backend"] = args.override_attention_backend
            if attention["backend"] == "flex_attention":
                attention.setdefault("flex_backend", args.flex_attention_backend)
        values = target.to_dict()
        target_depth = getattr(target, "num_target_layers", values["num_hidden_layers"])
        for key in (
            "model_type",
            "architectures",
            "auto_map",
            "layer_types",
            "composable_config",
            "block_size",
            "num_target_layers",
        ):
            values.pop(key, None)
        values["num_hidden_layers"] = len(recipe["backbone"]["layers"])
        config = ComposableDraftConfig(
            **values,
            composable_config=recipe,
            block_size=payload.get("block_size", args.block_size),
            num_target_layers=target_depth,
        )
        if args.target_layer_ids:
            config.composable_config["context"]["target_layer_ids"] = [
                int(value) for value in args.target_layer_ids.split(",")
            ]
        config.validate_recipe()
        return config

    def initialize(self, session):
        args = session.args
        source = args.resume_from or args.init_from
        if source:
            draft, loading = ComposableDraftModel.from_pretrained(
                source, torch_dtype=torch.bfloat16, output_loading_info=True
            )
            if any(
                loading.get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
            ):
                raise ValueError(
                    "Composable checkpoint weights do not match its model recipe: "
                    + str(loading)
                )
            draft = draft.cuda()
        else:
            draft = self.build_model(args)
        draft.requires_grad_(True)
        if (
            draft.config.composable_config["proposal"]["type"] == "selector"
            and args.selector_loss_alpha == 0
        ):
            draft.proposal.candidate_selector.requires_grad_(False)
        session.draft = draft
        args.block_size = draft.block_size
        args.num_draft_layers = draft.config.num_hidden_layers
        args.target_layer_ids = ",".join(map(str, draft.target_layer_ids))
        if (
            draft.config.composable_config["objective"]["type"] == "bv"
            and args.bv_sampling_provenance is None
        ):
            raise ValueError(
                "BV requires --bv-sampling-provenance target_full_vocab_t1; verify the dataset sampling provenance first"
            )

    def make_online(self, session):
        return ComposableTrainingModel(
            session.draft,
            session.components.lm_head,
            session.components.embed_tokens,
            session.args.mask_token_id,
            num_anchors=session.args.num_anchors,
            objective_chunk_blocks=session.args.objective_chunk_blocks,
        )

    def target_proxies(self, session):
        proxies = super().target_proxies(session)
        for proxy in proxies:
            proxy.method_config_key = "composable_config"
        return proxies

    def packet_spec(self, session, phase):
        from litespec.core.composable.packets import ComposablePacketSpec

        base = super().packet_spec(session, phase)
        return ComposablePacketSpec(
            base.batch_size,
            base.max_length,
            base.context_width,
            base.hidden_size,
            base.include_target_last_hidden,
        )

    def empty_packet(self, spec):
        from litespec.core.composable.packets import ComposableBatchPacket

        return ComposableBatchPacket.empty(spec)

    def produce_packet(self, session, phase, values, batch_id):
        from litespec.core.composable.packets import ComposableBatchPacket

        packet = super().produce_packet(session, phase, values, batch_id)
        if packet is None:
            return None
        return ComposableBatchPacket(
            packet.input_ids,
            packet.loss_mask,
            packet.target_context_hidden,
            packet.target_last_hidden,
            values[1].bool(),
        )

    def training_metadata(self, session):
        return {
            "loss_config": copy.deepcopy(
                session.draft.config.composable_config["objective"]
            ),
            "composable_recipe": copy.deepcopy(session.draft.config.composable_config),
            "selector_schedule_step": session.state.optimizer_step,
            "selector_effective_alpha": selector_alpha(
                session.args, session.state.optimizer_step, session.total_steps
            ),
            "selector_loss_alpha": session.args.selector_loss_alpha,
            "selector_warmup_ratio": session.args.selector_warmup_ratio,
            "selector_ramp_ratio": session.args.selector_ramp_ratio,
            "objective_chunk_blocks": session.args.objective_chunk_blocks,
            "num_anchors": session.args.num_anchors,
            "bv_sampling_provenance": session.args.bv_sampling_provenance,
        }

    def prepare_optimizer_window(self, session, first, iterator, count):
        recipe = session.draft.config.composable_config
        if recipe["proposal"]["type"] != "selector":
            return None
        objective = build_objective(recipe["objective"], recipe["proposal"]["type"])
        selector_objective = build_selector_objective(recipe)
        prepared, used = [], 0
        totals = torch.zeros(3, device="cuda", dtype=torch.float64)
        for index in range(count):
            raw = first if index == 0 else next(iterator)
            # Copy before advancing PacketReceiver: its slots are reusable.
            data = cpu_copy(raw)
            if session.topology is not None:
                ids, mask = data.input_ids, data.loss_mask
            else:
                ids, mask = data["input_ids"], data["loss_mask"]
                if session.tp_draft_rank is not None:
                    ids, mask = (
                        ids[session.tp_draft_rank : session.tp_draft_rank + 1],
                        mask[session.tp_draft_rank : session.tp_draft_rank + 1],
                    )
            layout = prepare_layout(
                ids, mask, session.draft.block_size, session.args.num_anchors
            )
            primary = objective.native_denominator(layout).item()
            auxiliary = selector_objective.native_denominator(layout).item()
            totals += totals.new_tensor([ids.size(0), primary, auxiliary])
            item = PreparedLogicalBatch(data, layout, totals)
            size = tensor_bytes(data)
            if used + size > session.args.window_buffer_mb * 1024**2:
                item.spill(Path(session.args.cache_dir) / "composable_windows")
            else:
                used += size
            prepared.append(item)
        if dist.is_initialized():
            dist.all_reduce(totals, group=session.process_group)
        if totals[1] <= 0 or totals[2] <= 0:
            for item in prepared:
                item.release()
            raise ValueError("Optimizer window has no supervised proposal positions")
        for item in prepared:
            item.window_denominators = totals
        return prepared

    def release_batch(self, data):
        if isinstance(data, PreparedLogicalBatch):
            data.release()

    def compute_step(
        self,
        session,
        data,
        *,
        microbatch_start=None,
        microbatch_end=None,
        collect_detailed_metrics=False,
    ):
        prepared = data if isinstance(data, PreparedLogicalBatch) else None
        data = prepared.materialize() if prepared is not None else data
        args, draft = session.args, session.draft
        if session.topology is not None:
            input_ids = data.input_ids[microbatch_start:microbatch_end].cuda()
            attention_mask = data.context_valid_mask[
                microbatch_start:microbatch_end
            ].cuda()
            loss_mask = data.loss_mask[microbatch_start:microbatch_end].cuda()
            target_hidden_states = data.target_context_hidden[
                microbatch_start:microbatch_end
            ].cuda()
            last = data.target_last_hidden
            last = (
                None if last is None else last[microbatch_start:microbatch_end].cuda()
            )
        else:
            from litespec.training.setup import select_tp_rank_batch

            input_ids = data["input_ids"].cuda(non_blocking=True)
            attention_mask = data["attention_mask"].cuda(non_blocking=True)
            loss_mask = data["loss_mask"].cuda(non_blocking=True)
            target_output = (
                None
                if session.target is None
                else session.target.generate_training_data(
                    input_ids, attention_mask, loss_mask, return_logits=False
                )
            )
            target_hidden_states, last = prepare_dflash_family_features(
                data, target_output, draft, True
            )
            if session.tp_draft_rank is not None:
                input_ids, attention_mask, loss_mask, target_hidden_states, last = [
                    select_tp_rank_batch(value, session.tp_draft_rank)
                    for value in (
                        input_ids,
                        attention_mask,
                        loss_mask,
                        target_hidden_states,
                        last,
                    )
                ]
        layout = (
            None
            if prepared is None
            else prepared.layout.batch_slice(microbatch_start, microbatch_end).to(
                input_ids.device
            )
        )
        return session.fsdp_model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=last,
            layout=layout,
            window_denominators=None
            if prepared is None
            else prepared.window_denominators,
            epoch_progress=session.state.phase_step / session.steps_per_epoch,
            selector_loss_alpha=selector_alpha(
                args, session.state.optimizer_step, session.total_steps
            ),
            collect_detailed_metrics=collect_detailed_metrics,
            context_valid_mask=attention_mask.bool(),
        )


ADAPTER = ComposableTrainingAdapter("composable", "ComposableDraftModel")
