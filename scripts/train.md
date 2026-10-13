## DLite training recipe

The three DLite training modes now have YAML configurations. Edit the matching  
file, then launch it through its existing shell entrypoint:

```plaintext
bash scripts/dflash2/run_training_dflash2.sh scripts/config/dflash2.yaml
bash scripts/dlite/run_training_dlite_teacher.sh scripts/config/dlite_teacher.yaml
bash scripts/dlite/run_training_dlite_two_stage.sh scripts/config/dlite_two_stage.yaml
```

Each YAML has a `launcher` section for paths and torchrun settings, plus an  
`arguments` section whose keys map directly to training CLI options. Set  
exactly one of `launcher.train_data` and `launcher.train_hidden_states`.

All YAML-launched workflows (DLite SFT/teacher/two-stage, DFlash, DFlash2,
and DSpark) save `training_config.yaml` beside the model in every checkpoint
directory, including final and phase-boundary saves. This is the original YAML
captured when the launcher reads it, preserving comments and formatting; later
edits or deletion of the source file do not affect it. CLI overrides are not
merged into this source snapshot. Direct CLI runs without a YAML launcher do
not create a source YAML snapshot. A resumed YAML launch records its new launch
snapshot in the new checkpoints.

Use `launcher.train_data` for online target-model prefill, or  
`launcher.train_hidden_states` for offline training. Set only one. Offline mode  
requires `tp_size: 1` and `shard_draft_by_tp: false`.

2>&1 &

```plaintext
export WANDB_MODE=offline
/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/stop_keeper.sh

cd /inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/LiteSpec
source .venv/bin/activate

bash scripts/dlite/run_training_dlite_sft.sh scripts/config/dlite_sft.yaml > "whz_mtp_logs/train_dlite_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# ========== 关键修改：等待训练完成 ==========
TRAIN_PID=$!
echo "训练已启动，PID: $TRAIN_PID，等待训练完成..."

if wait "$TRAIN_PID"; then
  TRAIN_EXIT=0
else
  TRAIN_EXIT=$?
fi

echo "训练已结束，退出码: $TRAIN_EXIT"

/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/start_keeper.sh

exit "$TRAIN_EXIT"
```
