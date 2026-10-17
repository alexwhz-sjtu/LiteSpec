"""Self-contained synthetic test fixture, not a converted user dataset."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from litespec.data import build_training_dataset
from datasets import Dataset

ROOT = Path(__file__).resolve().parent
out = ROOT / "fixtures"
out.mkdir(exist_ok=True)
target = out / "target"
target.mkdir(exist_ok=True)
torch.manual_seed(123)
vocab = {
    token: i
    for i, token in enumerate(
        ["<pad>", "<unk>", "<eos>", "<|im_start|>", "<|im_end|>"]
        + ["user", "assistant", "system", "You", "are", "a", "helpful", "assistant."]
        + [f"t{i}" for i in range(115)]
    )
}
tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    pad_token="<pad>",
    unk_token="<unk>",
    eos_token="<eos>",
    additional_special_tokens=["<|im_start|>", "<|im_end|>"],
)
tokenizer.save_pretrained(target)
config = Qwen3Config(
    vocab_size=len(vocab),
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=512,
    tie_word_embeddings=False,
)
model = Qwen3ForCausalLM(config).eval()
model.save_pretrained(target)
records = [
    {
        "conversations": [
            {"role": "user", "content": f"t{i} t1"},
            {
                "role": "assistant",
                "content": " ".join(f"t{(i + j) % 100}" for j in range(24 + i % 3)),
            },
        ]
    }
    for i in range(10)
]
(out / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
recipe = {
    "context": {"type": "dflare", "target_layer_ids": [1, 4]},
    "backbone": {
        "type": "transformer",
        "layers": [
            {
                "attention": {
                    "representation": "mha",
                    "access": "dense",
                    "backend": "sdpa",
                }
            },
            {
                "attention": {
                    "representation": "mla",
                    "access": "sliding_window",
                    "window_size": 8,
                    "backend": "sdpa",
                    "kv_lora_rank": 8,
                    "qk_nope_head_dim": 4,
                    "qk_rope_head_dim": 4,
                    "v_head_dim": 8,
                },
                "local_operator": {
                    "type": "dflash2_conv",
                    "kernel_size": 2,
                    "group_size": 8,
                },
            },
        ],
    },
    "proposal": {"type": "selector", "rank": 8, "top_k": 5},
    "objective": {"type": "dpace", "alpha": 0.5},
}
(out / "draft.json").write_text(
    json.dumps({"block_size": 4, "composable_config": recipe}, indent=2)
)
processed = build_training_dataset(
    dataset=Dataset.from_list(records),
    tokenizer=tokenizer,
    chat_template="qwen",
    max_length=128,
    is_preformatted=False,
    num_proc=1,
)
cache = out / "offline"
cache.mkdir(exist_ok=True)
metadata = []
archive_path = cache / "shard-00000.tar"
with tarfile.open(archive_path, "w") as archive:
    for index, row in enumerate(processed):
        ids = torch.as_tensor(row["input_ids"]).reshape(-1)
        mask = torch.as_tensor(row["loss_mask"]).reshape(-1)
        with torch.no_grad():
            result = model.model(ids[None], output_hidden_states=True, use_cache=False)
        payload = {
            "schema_version": 2,
            "sample_index": index,
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "loss_mask": mask,
            "hidden_states": {
                i: result.hidden_states[i + 1][0].to(torch.bfloat16) for i in range(6)
            },
            "final_hidden_state": result.hidden_states[-1][0].to(torch.bfloat16),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        content = buffer.getvalue()
        name = f"samples/{index:012d}.pt"
        member = tarfile.TarInfo(name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
        active = mask.nonzero().flatten().tolist()
        runs = []
        for position in active:
            if runs and position == runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([position, position + 1])
        metadata.append(
            {
                "sequence_length": len(ids),
                "supervised_token_count": len(active),
                "loss_mask_runs": runs,
            }
        )
manifest = {
    "schema_version": 2,
    "save_mode": "regen_full",
    "complete": True,
    "sample_index_version": 1,
    "target_model": str(target),
    "num_hidden_layers": 6,
    "hidden_size": 32,
    "max_length": 128,
    "final_norm_layer_id": 5,
    "stored_hidden_layer_ids": list(range(6)),
    "successful_samples": len(metadata),
    "shards": [
        {
            "path": archive_path.name,
            "success_start": 0,
            "success_end": len(metadata) - 1,
            "sample_count": len(metadata),
            "source_indices": list(range(len(metadata))),
            "sample_metadata": metadata,
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        }
    ],
}
(cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("FIXTURES", out, "records", len(metadata), "vocab", len(vocab))
