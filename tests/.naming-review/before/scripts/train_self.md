## DLite training recipe

The three DLite training modes now have YAML configurations. Edit the matching
file, then launch it through its existing shell entrypoint:

```bash
bash scripts/dlite/run_training_dlite_sft.sh scripts/config/dlite_sft.yaml
bash scripts/dlite/run_training_dlite_teacher.sh scripts/config/dlite_teacher.yaml
bash scripts/dlite/run_training_dlite_two_stage.sh scripts/config/dlite_two_stage.yaml
```

Each YAML has a `launcher` section for paths and torchrun settings, plus an
`arguments` section whose keys map directly to training CLI options. Set
exactly one of `launcher.train_data` and `launcher.train_hidden_states`.

Use `launcher.train_data` for online target-model prefill, or
`launcher.train_hidden_states` for offline training. Set only one. Offline mode
requires `tp_size: 1` and `shard_draft_by_tp: false`.

### single node

```bash
cd /path/to/LiteSpec
source .venv/bin/activate

bash scripts/dlite/run_training_dlite_sft.sh scripts/config/dlite_sft.yaml
```

```bash
cd /path/to/LiteSpec

NODE_JIT_CACHE="/tmp/flashmtp-jit-$(hostname)"
export TVM_FFI_CACHE_DIR="${NODE_JIT_CACHE}/tvm-ffi"
export TORCHINDUCTOR_CACHE_DIR="${NODE_JIT_CACHE}/torchinductor"
export TRITON_CACHE_DIR="${NODE_JIT_CACHE}/triton"
source .venv/bin/activate

bash scripts/dlite/run_training_dlite_sft.sh \
  scripts/config/dlite_sft.yaml \
  --max-length 10240 --num-anchors 768 \
  > "whz_mtp_logs/train_dlite_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# ========== 关键修改：等待训练完成 ==========
TRAIN_PID=$!
echo "训练已启动，PID: $TRAIN_PID，等待训练完成..."

if wait "$TRAIN_PID"; then
  TRAIN_EXIT=0
else
  TRAIN_EXIT=$?
fi

echo "训练已结束，退出码: $TRAIN_EXIT"


exit "$TRAIN_EXIT"

```
```bash
cd /data/wanghanzhen/projects/SpecDecoding/LiteSpec

bash scripts/dflash_family/run_training_dflash2.sh \
  scripts/config/dflash2.yaml
```
