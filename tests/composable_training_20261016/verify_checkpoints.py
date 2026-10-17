import json
import math
import re
from pathlib import Path
import torch
from safetensors import safe_open
torch.set_num_threads(4)
root = Path(__file__).resolve().parent
results = []
for case in ("default", "heterogeneous"):
    final = root / ("output_" + case) / "final"
    if not final.exists():
        continue
    state = torch.load(final / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["optimizer_step"] == 500, state["optimizer_step"]
    assert state["cursor"]["global_step"] == 500, state["cursor"]
    assert state["distributed_training_state"]["world_size"] == 8
    shards = list(final.glob("training_state_rank_*.pt"))
    assert len(shards) == 8, len(shards)
    for shard in shards:
        local = torch.load(shard, map_location="cpu", weights_only=False)
        assert local["optimizer_step"] == 500
        assert local["cursor"]["global_step"] == 500
        for param in local.get("fp32_master_params", []):
            if isinstance(param, torch.Tensor):
                assert torch.isfinite(param).all()
        del local
    total_parameters = 0
    for weights in final.glob("*.safetensors"):
        with safe_open(weights, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor=f.get_tensor(key)
                assert torch.isfinite(tensor).all(), key
                total_parameters += tensor.numel()
    assert total_parameters > 0
    log=(root/(case+".log")).read_text(errors="replace")
    losses=re.findall(r"composable optimizer_step=(\d+) micro_step=\d+ loss=([^\s]+)", log)
    assert len(losses) == 10
    assert all(math.isfinite(float(loss)) for _, loss in losses)
    results.append({
        "case": case, "optimizer_step": state["optimizer_step"],
        "global_step": state["cursor"]["global_step"],
        "world_size": 8, "rank_checkpoints": len(shards),
        "finite_model_parameters": total_parameters,
        "loss_at_50": float(losses[0][1]), "loss_at_500": float(losses[-1][1]),
        "checkpoint": str(final),
        "scheduler_total_steps": state["scheduler_total_steps"],
        "recipe": state["composable_recipe"],
    })
    print(json.dumps({k:v for k,v in results[-1].items() if k!="recipe"}), flush=True)
(root / "verified_checkpoints.json").write_text(json.dumps(results, indent=2))
