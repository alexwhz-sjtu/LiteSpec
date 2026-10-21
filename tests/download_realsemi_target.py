"""Fetch and validate the fixed target snapshot used by all three experiments."""

import json
from pathlib import Path
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

root = Path("/data/temp/artifacts/models/Qwen3-4B")
api = HubApi()
info = api.get_model("Qwen/Qwen3-4B")
# SDK snapshot metadata records the resolved revision even when the public branch is used.
path = snapshot_download("Qwen/Qwen3-4B", local_dir=str(root))
index = json.loads((root / "model.safetensors.index.json").read_text())
for name in set(index["weight_map"].values()):
    assert (root / name).is_file(), name
for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    assert (root / name).is_file(), name
metadata = {"model_id": "Qwen/Qwen3-4B", "local_path": str(root), "model_info": info}
(root / "realsemi_source.json").write_text(
    json.dumps(metadata, ensure_ascii=False, default=str, indent=2)
)
print("TARGET_READY", path, flush=True)
