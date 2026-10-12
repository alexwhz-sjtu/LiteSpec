## DLite training recipe

The three DLite training modes now have YAML configurations. Edit the matching
file, then launch it through its existing shell entrypoint:

```bash
bash scripts/dflash2/run_training_dflash2.sh scripts/config/dflash2.yaml
bash scripts/dlite/run_training_dlite_teacher.sh scripts/config/dlite_teacher.yaml
bash scripts/dlite/run_training_dlite_two_stage.sh scripts/config/dlite_two_stage.yaml
```

Each YAML has a `launcher` section for paths and torchrun settings, plus an
`arguments` section whose keys map directly to training CLI options. Set
exactly one of `launcher.train_data` and `launcher.train_hidden_states`.

Use `launcher.train_data` for online target-model prefill, or
`launcher.train_hidden_states` for offline training. Set only one. Offline mode
requires `tp_size: 1` and `shard_draft_by_tp: false`.
