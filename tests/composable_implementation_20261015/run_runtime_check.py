"""Reproducible short checks through the real public training entry point."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "case", choices=["online", "offline", "fsdp", "disagg", "tp", "resume"]
)
parser.add_argument(
    "--objective", choices=["hard_ce", "soft_ce", "forward_kl", "tv", "nla", "dpace"]
)
parser.add_argument(
    "--proposal", choices=["parallel", "markov", "rnn", "sequential_rnn", "selector"]
)
args = parser.parse_args()
case = args.case
label = (
    case
    + ("_" + args.objective if args.objective else "")
    + ("_" + args.proposal if args.proposal else "")
)
recipe_path = ROOT / "fixtures/draft.json"
if args.objective or args.proposal:
    recipe = json.loads(recipe_path.read_text())
    if args.objective:
        recipe["composable_config"]["objective"] = {
            "type": args.objective,
            "loss_decay_gamma": 2.0,
        }
    if args.proposal:
        recipe["composable_config"]["proposal"]["type"] = args.proposal
    recipe_path = ROOT / "fixtures" / ("draft_" + label + ".json")
    recipe_path.write_text(json.dumps(recipe, indent=2))

devices = {
    "online": "0",
    "offline": "1",
    "fsdp": "0,1",
    "disagg": "2,3",
    "tp": "4,5",
    "resume": "6",
}[case]
ranks = 2 if case in ("fsdp", "disagg", "tp") else 1
command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    f"--nproc_per_node={ranks}",
    "-m",
    "scripts.composable.train",
    "--target-model-path",
    str(ROOT / "fixtures/target"),
    "--draft-config",
    str(recipe_path),
    "--output-dir",
    str(ROOT / "runs" / (label + "_verified")),
    "--max-length",
    "128",
    "--num-anchors",
    "2",
    "--objective-chunk-blocks",
    "1",
    "--accumulation-steps",
    "2",
    "--num-epochs",
    "1",
    "--mask-token-id",
    "0",
    "--dataloader-num-workers",
    "0",
    "--build-dataset-num-proc",
    "1",
    "--log-interval",
    "1",
    "--save-interval",
    "2",
    "--cache-dir",
    str(ROOT / ("cache_" + case)),
    "--dist-timeout",
    "180",
]
if case == "offline":
    command += ["--train-hidden-states-path", str(ROOT / "fixtures/offline")]
else:
    command += ["--train-data-path", str(ROOT / "fixtures/data.jsonl")]
if case == "disagg":
    command += [
        "--disaggregate",
        "--target-ranks-per-node",
        "1",
        "--draft-ranks-per-node",
        "1",
        "--node-batch-size",
        "2",
        "--draft-micro-batch-size",
        "1",
        "--pipeline-depth",
        "2",
    ]
if case == "tp":
    command += [
        "--target-model-backend",
        "sglang",
        "--tp-size",
        "2",
        "--batch-size",
        "2",
        "--shard-draft-by-tp",
        "--sglang-attention-backend",
        "torch_native",
        "--sglang-mem-fraction-static",
        "0.1",
    ]
if case == "resume":
    command += [
        "--resume-from",
        str(
            Path(json.loads((ROOT / "online_result.json").read_text())["output_dir"])
            / "epoch_0_step_4"
        ),
    ]
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=devices,
    PYTHONPATH=str(PROJECT),
    OMP_NUM_THREADS="1",
    TORCHINDUCTOR_CACHE_DIR=str(ROOT / "inductor_cache"),
    TRITON_CACHE_DIR=str(ROOT / "triton_cache"),
)
log = ROOT / (label + "_verified.log")
print("RUN", case, "GPU", devices, flush=True)
with log.open("w") as stream:
    result = subprocess.run(
        command,
        cwd=PROJECT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
output_match = __import__("re").search(
    r"Training output directory: (.+)", log.read_text()
)
record = {
    "output_dir": output_match.group(1) if output_match else None,
    "case": case,
    "returncode": result.returncode,
    "command": command,
    "log": str(log),
}
(ROOT / (label + "_result.json")).write_text(json.dumps(record, indent=2))
print("RESULT", case, result.returncode, flush=True)
if result.returncode:
    print(log.read_text()[-14000:])
raise SystemExit(result.returncode)
