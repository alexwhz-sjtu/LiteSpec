"""Routing/layout unit tests; actual NCCL integration lives in the smoke script."""

import pytest
import torch
import torch.distributed as dist

from litespec.training.fsdp_layout import ParameterLayout, ParameterSpec
from litespec.training.muon import ShardedMuon


def test_layout_rejects_gaps_overlaps_and_negative_lengths():
    spec = ParameterSpec("fc.weight", (3, 5), "muon")
    for shards in (((0, 6), (7, 8)), ((0, 10), (8, 7)), ((0, 16), (0, -1))):
        with pytest.raises(ValueError):
            ParameterLayout(spec, shards)
    assert ParameterLayout(spec, ((0, 0), (0, 15))).shards[0][1] == 0


def _routing_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(23)
        full = [torch.randn(3, 5) for _ in range(5)]
        # Empty shards, non-row-aligned splits, and a partial batch.
        splits = [
            ((0, 0), (0, 15)),
            ((0, 7), (7, 8)),
            ((0, 15), (0, 0)),
            ((0, 3), (3, 12)),
            ((0, 8), (8, 7)),
        ]
        layouts = [
            ParameterLayout(ParameterSpec(str(i), (3, 5), "muon"), s)
            for i, s in enumerate(splits)
        ]
        params = [
            torch.nn.Parameter(x.flatten()[s[rank][0] : sum(s[rank])].clone())
            for x, s in zip(full, splits)
        ]
        refs = [torch.nn.Parameter(x.clone()) for x in full]
        opt = ShardedMuon(params, layouts, matrices_per_rank=2)
        ref = torch.optim.Muon(refs, lr=0.02, momentum=0.95, weight_decay=0.1)
        for step in range(4):
            for i, (p, q, shards) in enumerate(zip(params, refs, splits)):
                g = torch.randn_like(q)
                offset, length = shards[rank]
                p.grad = (
                    g.flatten()[offset : offset + length].clone() if length else None
                )
                q.grad = g
                if step == 2 and i == 2:
                    p.grad = q.grad = None
            opt.step()
            ref.step()
            for p, q, shards in zip(params, refs, splits):
                offset, length = shards[rank]
                torch.testing.assert_close(
                    p, q.flatten()[offset : offset + length], rtol=0, atol=5e-4
                )
                if length:
                    torch.testing.assert_close(
                        opt.state[p]["momentum_buffer"],
                        ref.state[q]["momentum_buffer"].flatten()[
                            offset : offset + length
                        ],
                        rtol=0,
                        atol=0,
                    )
    finally:
        dist.destroy_process_group()


def test_two_rank_routing(tmp_path):
    torch.multiprocessing.spawn(
        _routing_worker, args=((tmp_path / "muon_init").as_uri(),), nprocs=2
    )
