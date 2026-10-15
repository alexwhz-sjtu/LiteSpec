## DLite training recipe

The three DLite training modes now have YAML configurations. Edit the matching  
file, then launch it through its existing shell entrypoint:

```plaintext
bash scripts/dflash2/run_training_dflash2.sh scripts/config/dflash2.yaml
bash scripts/dlite/run_training_dlite_teacher.sh scripts/config/dlite_teacher.yaml
bash scripts/dlite/run_training_dlite_two_stage.sh scripts/config/dlite_two_stage.yaml
bash scripts/composable/run_training_composable.sh scripts/config/composable.yaml 
```

Each YAML has a `launcher` section for paths and torchrun settings, plus an  
`arguments` section whose keys map directly to training CLI options. Set  
exactly one of `launcher.train_data` and `launcher.train_hidden_states`.

Set `arguments.weight_decay` (default `0.0`) to control AdamW weight decay
when `use_muon: false`. For example, append `--no-use-muon --weight-decay 0.1`
to a training command. With Muon enabled, this field is ignored; use
`muon_weight_decay` and `muon_adamw_weight_decay` for the two parameter groups.

`bash tests/run_dlite_optimizer_sweep.sh --dry-run` previews the five DLite
optimizer experiments without starting training or keeper scripts. Remove
`--dry-run` to run them sequentially, stopping on the first failure and restoring
the keeper when the queue exits. Keeper scripts default to the repository's
parent directory (`KEEPER_DIR` overrides it). Set a fresh `SWEEP_ID` for each
queue; all nodes of a distributed job must use the same ID and experiment order.
Each experiment has separate model output and logs. Experiment 5 inherits its
learning rate from the baseline YAML, not from experiment 2.

All YAML-launched workflows (DLite SFT/teacher/two-stage, DFlash, DFlash2,
and DSpark) save `training_config.yaml` beside the model in every checkpoint
directory, including final and phase-boundary saves. It contains the launch-time
YAML with command-line overrides merged in using the parsed argument types,
including `--no-*` flags, output paths and W&B names. Torchrun settings also
reflect `PET_*` environment overrides. The source file is not modified; the
saved YAML is reserialized, so original comments and formatting are not retained.
Later edits or deletion of the source file do not affect the captured config.
Direct CLI runs without a YAML launcher do not create this YAML snapshot. A
resumed YAML launch records its new merged launch configuration.

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

