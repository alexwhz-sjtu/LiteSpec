"""Check runtime backend injection and Flex parity for all representation/access pairs."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from test_implementation import config, fixture
from transformers import Qwen3Config

from litespec.core.composable.batch import prepare_layout
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.modeling.draft.composable import ComposableDraftModel
from litespec.training.recipes.composable import ComposableTrainingAdapter

root = Path(__file__).resolve().parent
injection_checks = []
target = Qwen3Config(
    hidden_size=32,
    intermediate_size=48,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=8,
    vocab_size=23,
)
for path in sorted(Path("configs").glob("*.json")):
    payload = json.loads(path.read_text())
    for layer in payload["composable_config"]["backbone"]["layers"]:
        assert "backend" not in layer["attention"]
        assert "flex_backend" not in layer["attention"]
    for backend in ("eager", "sdpa", "flex_attention"):
        args = SimpleNamespace(
            draft_config=str(path),
            attention_backend=backend,
            override_attention_backend=None,
            flex_attention_backend="TRITON",
            block_size=8,
            target_layer_ids=None,
        )
        resolved = ComposableTrainingAdapter(
            "composable", "ComposableDraftModel"
        ).build_config(args, source_config=target)
        for layer in resolved.composable_config["backbone"]["layers"]:
            assert layer["attention"]["backend"] == backend
            if backend == "flex_attention":
                assert layer["attention"]["flex_backend"] == "TRITON"
        injection_checks.append({"recipe": str(path), "backend": backend})

records = []
for context in ("dflash", "dflare"):
    for representation in ("mha", "mla"):
        for access in ("dense", "sliding_window"):
            torch.manual_seed(105)
            cfg = config(context=context)
            cfg.head_dim = 16
            for layer in cfg.composable_config["backbone"]["layers"]:
                attention = layer["attention"]
                attention.update(representation=representation, access=access)
                if access == "sliding_window":
                    attention["window_size"] = 3
                if representation == "mla":
                    attention.update(
                        kv_lora_rank=8,
                        qk_nope_head_dim=8,
                        qk_rope_head_dim=8,
                        v_head_dim=16,
                    )
            base, embedding, head, ids, hidden, teacher, mask = fixture(cfg)
            other_cfg = copy.deepcopy(cfg)
            for layer in other_cfg.composable_config["backbone"]["layers"]:
                layer["attention"].update(
                    backend="flex_attention", flex_backend="TRITON"
                )
            flex = ComposableDraftModel(other_cfg)
            flex.load_state_dict(base.state_dict())
            base, flex = base.cuda(), flex.cuda()
            embedding, head = embedding.cuda(), head.cuda()
            ids, hidden, teacher, mask = [
                value.cuda() for value in (ids, hidden, teacher, mask)
            ]
            layout = prepare_layout(ids, mask, 4, 2)
            losses = []
            for model in (base, flex):
                wrapper = ComposableTrainingModel(
                    model, head, embedding, 0, num_anchors=2, objective_chunk_blocks=1
                )
                output = wrapper(ids, hidden, mask, teacher, layout=layout)
                losses.append(output.loss.detach())
                output.loss.backward()
            torch.testing.assert_close(losses[0], losses[1], atol=2e-5, rtol=2e-5)
            max_error = 0.0
            for (name, p), (_, q) in zip(
                base.named_parameters(), flex.named_parameters()
            ):
                assert p.grad is not None and q.grad is not None, name
                torch.testing.assert_close(
                    p.grad, q.grad, atol=2e-4, rtol=2e-3, msg=name
                )
                max_error = max(max_error, (p.grad - q.grad).abs().max().item())
            record = dict(
                context=context,
                representation=representation,
                access=access,
                loss_error=(losses[0] - losses[1]).abs().item(),
                max_gradient_error=max_error,
            )
            records.append(record)
            print(json.dumps(record), flush=True)

(root / "backend_matrix_result.json").write_text(
    json.dumps(
        {
            "runtime_injection": injection_checks,
            "flex_vs_sdpa": records,
            "torch_version": torch.__version__,
        },
        indent=2,
    )
    + "\n"
)
print("BACKEND_MATRIX_OK", flush=True)
