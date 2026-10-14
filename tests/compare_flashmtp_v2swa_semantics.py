"""CPU audit of selected semantics in the two checked-out training frameworks.

Extract actual tensor-only definitions with AST to avoid initializing either
repository's CUDA/SGLang runtime. This is not an end-to-end equivalence test.
Run: .venv/bin/python tests/compare_flashmtp_v2swa_semantics.py
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "FlashMTP_v2swa"


def extract(path, name, *, owner=None, extra=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    namespace = {"torch": torch, "nn": torch.nn, **(extra or {})}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    result = {}
    old_head_path = OLD / "specforge/modeling/draft/flashmtp_markov_head.py"
    spec = importlib.util.spec_from_file_location("audit_old_head", old_head_path)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    new_head = extract(ROOT / "litespec/modeling/draft/dlite.py", "DLiteSequentialHead", extra={"SEQUENTIAL_HEAD_TYPES": ("rnn",)})
    common = dict(vocab_size=11, hidden_size=6, max_prediction_length=3)
    old = old_module.FlashMTPMarkovHead(head_type="rnn_easy", markov_rank=4, markov_output_mode="direct", **common)
    new = new_head(head_type="rnn", sequential_rank=4, **common)
    new.load_state_dict(old.state_dict())
    hidden_old = torch.randn(2, 3, 6, requires_grad=True)
    hidden_new = hidden_old.detach().clone().requires_grad_()
    ids = torch.randint(11, (2, 3))
    out_old = old.project_logits(old.forward_teacher_forcing(hidden_states=hidden_old, prev_token_ids=ids, output_mode="direct"))
    out_new = new.project_logits(new.forward_teacher_forcing(hidden_states=hidden_new, prev_token_ids=ids))
    out_old.square().sum().backward()
    out_new.square().sum().backward()
    differences = [(a.grad - b.grad).abs().max().item() for a, b in zip(old.parameters(), new.parameters())]
    result["rnn_easy_direct_vs_dlite_rnn"] = dict(logits_max_error=(out_old-out_new).abs().max().item(), input_grad_max_error=(hidden_old.grad-hidden_new.grad).abs().max().item(), parameter_grad_max_error=max(differences))
    assert max(result["rnn_easy_direct_vs_dlite_rnn"].values()) == 0

    model_old = OLD / "specforge/modeling/draft/flashmtp.py"
    pos_old = extract(model_old, "build_block_position_ids", owner="FlashMTPDraftModel")
    query_pos = extract(model_old, "build_draft_query_position_ids", owner="FlashMTPDraftModel")
    old_stub = SimpleNamespace(local_position=True, core_draft_query_length=4, condition_slot_count=2, pivot_query_embedding=False)
    old_stub.build_draft_query_position_ids = lambda *a, **kw: query_pos(old_stub, *a, **kw)
    anchor = torch.tensor([[5]])
    empty = torch.empty(1, 1, 0, dtype=torch.long)
    op = pos_old(old_stub, anchor, empty, empty.bool())
    pos_new = extract(ROOT / "litespec/modeling/draft/dlite.py", "build_block_position_ids", owner="DLiteDraftModel")
    np = pos_new(SimpleNamespace(is_teacher=False, chs_num_layers=2, block_size=4), anchor, anchor.unsqueeze(-1), torch.ones(1, 1, 1, dtype=torch.bool))
    result["rope_window_one"] = dict(old_chs=op[0].tolist(), old_queries=op[1].tolist(), new_chs=np[0].tolist(), new_queries=np[1].tolist())
    assert op[1].tolist() == np[1].tolist() == [[1, 2, 3, 4]]
    assert op[0].tolist() == np[0].tolist() == [[0, 0]]

    def dense_block_mask(mod, *, B, H, Q_LEN, KV_LEN, device):
        b = torch.arange(B).view(B, 1, 1, 1)
        q = torch.arange(Q_LEN).view(1, 1, Q_LEN, 1)
        kv = torch.arange(KV_LEN).view(1, 1, 1, KV_LEN)
        return mod(b, 0, q, kv)

    old_dflash = OLD / "specforge/core/dflash.py"
    old_mask = extract(old_dflash, "create_dflash_block_mask", extra={"create_block_mask": dense_block_mask})
    new_mask = extract(ROOT / "litespec/core/attention_masks.py", "create_dflash_sdpa_mask")
    mask_args = dict(anchor_positions=anchor, block_keep_mask=torch.tensor([[True]]), S=10, block_size=4, device="cpu")
    mask_old, mask_full, mask_sliding = old_mask(**mask_args), new_mask(**mask_args), new_mask(**mask_args, sliding_window=4)
    assert torch.equal(mask_old, mask_full)
    assert not torch.equal(mask_old, mask_sliding)
    result["dflash_masks"] = dict(full_equal=True, old_first_query_visible_keys=mask_old[0, 0, 0].nonzero().flatten().tolist(), sliding_first_query_visible_keys=mask_sliding[0, 0, 0].nonzero().flatten().tolist())

    loss_mask = torch.ones(1, 10)
    loss_mask[:, 0] = 0
    stub = SimpleNamespace(block_size=4, num_anchors=100)
    old_sample = extract(old_dflash, "_sample_anchor_positions", owner="OnlineDFlashModel")
    new_sample = extract(ROOT / "litespec/core/_dflash_common.py", "_sample_anchor_positions", owner="DFlashTrainingBase")
    old_a, _ = old_sample(stub, 10, loss_mask, torch.device("cpu"))
    new_a, _ = new_sample(stub, 10, loss_mask, torch.device("cpu"))
    result["dflash_sampled_anchors"] = dict(old=old_a.tolist(), new=new_a.tolist())
    assert new_a.max() > 10-stub.block_size

    sorted_anchors = extract(ROOT / "litespec/core/block_training.py", "sample_sorted_anchors")
    old_sample = extract(OLD / "specforge/core/flashmtp.py", "_sample_anchor_positions", owner="OnlineFlashMTPModel")
    new_sample = extract(ROOT / "litespec/core/dlite.py", "sample_anchor_positions", owner="OnlineDLiteModel", extra={"sample_sorted_anchors": sorted_anchors})
    stub.draft_model = SimpleNamespace(draft_query_length=4, chs_num_layers=2)
    stub.flex_attention_backend = "TRITON"
    old_a, _ = old_sample(stub, 10, loss_mask, torch.device("cpu"))
    new_a, _ = new_sample(stub, 10, loss_mask)
    result["flashmtp_vs_dlite_sampled_anchors"] = dict(old=old_a.tolist(), new=new_a.tolist())
    assert old_a.shape[1] == 5 and new_a.shape[1] == 6

    # Direct mathematical witness for the two normalization definitions.
    w = torch.tensor([1., 1.], requires_grad=True)
    numerators, denominators = w * torch.tensor([1., 27.]), torch.tensor([1., 9.])
    old_loss = (numerators / denominators).mean()
    new_loss = numerators.sum() / denominators.sum()
    result["normalization_witness"] = dict(old_loss=old_loss.item(), new_loss=new_loss.item(), old_gradient=torch.autograd.grad(old_loss, w, retain_graph=True)[0].tolist(), new_gradient=torch.autograd.grad(new_loss, w)[0].tolist())
    assert old_loss.item() != new_loss.item()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
