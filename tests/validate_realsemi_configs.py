"""Validate public launch/config contracts without allocating model weights."""

import hashlib
import json
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.config.training_launcher import arguments_to_cli
from litespec.training.algorithms import get_training_adapter

root = Path(__file__).resolve().parents[1]
manifest_path = root / "tests/realsemi_experiments.json"
manifest = json.loads(manifest_path.read_text())
for item in manifest["experiments"]:
    cfg = yaml.safe_load(Path(item["config"]).read_text())
    cli = arguments_to_cli(cfg["arguments"]) + [
        "--target-model-path",
        cfg["launcher"]["target_model"],
        "--train-data-path",
        cfg["launcher"]["train_data"],
        "--output-dir",
        cfg["launcher"]["output_dir"],
    ]
    adapter = get_training_adapter(item["algorithm"])
    args = adapter.parse_args(argv=cli)
    config = adapter.build_config(args)
    assert config.num_hidden_layers == 5
    assert config.block_size == 8
    assert args.num_epochs == 6 and args.log_interval == 50 and args.use_muon
    if item["algorithm"] == "realsemi":
        assert config.realsemi_config["num_parallel_layers"] == 4
        assert config.realsemi_config["ce_loss_alpha"] == 0.1
        assert config.realsemi_config["l1_loss_alpha"] == 0.9
    print("CONFIG_PASSED", item["name"])
model = Path("/data/temp/artifacts/models/Qwen3-4B")
source_path = model / "realsemi_source.json"
source = json.loads(source_path.read_text())
source["snapshot_revision"] = (model / ".mv").read_text()
source["file_sha256"] = {}
for file in sorted(model.glob("*.safetensors")):
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    source["file_sha256"][file.name] = digest.hexdigest()
source_path.write_text(json.dumps(source, ensure_ascii=False, default=str, indent=2))
manifest["target_source"] = {
    k: source[k] for k in ("model_id", "snapshot_revision", "file_sha256")
}
digest = hashlib.sha256()
with open(manifest["data_path"], "rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
manifest["data_sha256"] = digest.hexdigest()
manifest_path.write_text(json.dumps(manifest, indent=2))
print("TARGET_AND_DATA_FINGERPRINTED")
