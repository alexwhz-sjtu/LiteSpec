# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: framework_integration_baseline
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: scripts/config/training_launcher.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/scripts/config/training_launcher.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Runtime YAML/JSON path reuse; do not hand nested dictionaries to flat arguments_to_cli.


# BEGIN load_yaml_config — upstream lines 57-97
def load_yaml_config(
    path: Path,
    *,
    identity_key: str,
    identity_value: str,
    extra_launcher_keys: Sequence[str] = (),
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    """Parse and snapshot the same YAML read, preserving comments and newlines."""
    if not path.is_file():
        raise ConfigError(f"config does not exist: {path}")
    try:
        source = path.read_bytes().decode("utf-8")
        config = yaml.safe_load(source)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read {path}: {exc}") from exc
    config = _mapping(config, "config")
    unknown = sorted(set(config) - {identity_key, "launcher", "arguments"})
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")
    configured_identity = config.get(identity_key)
    if configured_identity != identity_value:
        raise ConfigError(
            f"config {identity_key} {configured_identity!r} does not match "
            f"launcher {identity_value!r}"
        )

    launcher = _mapping(config.get("launcher"), "launcher")
    allowed = COMMON_LAUNCHER_KEYS | set(extra_launcher_keys)
    unknown = sorted(set(launcher) - allowed)
    if unknown:
        raise ConfigError(f"unknown launcher keys: {unknown}")
    missing_field_names = [key for key in ("target_model", "output_dir") if not launcher.get(key)]
    if missing_field_names:
        raise ConfigError(f"missing launcher keys: {missing_field_names}")
    if bool(launcher.get("train_data")) == bool(launcher.get("train_hidden_states")):
        raise ConfigError(
            "set exactly one of launcher.train_data and launcher.train_hidden_states"
        )

    arguments = _mapping(config.get("arguments", {}), "arguments")
    return launcher, arguments, source

# END load_yaml_config

# BEGIN arguments_to_cli — upstream lines 211-236
def arguments_to_cli(arguments: Mapping[str, Any]) -> list[str]:
    cli: list[str] = []
    for raw_name, value in arguments.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigError("argument names must be non-empty strings")
        option = "--" + raw_name.strip().replace("_", "-")
        if value is None:
            continue
        if value is False:
            cli.append("--no-" + raw_name.strip().replace("_", "-"))
            continue
        cli.append(option)
        if value is True:
            continue
        if isinstance(value, list):
            if not value:
                raise ConfigError(f"{raw_name} list must not be empty")
            cli.extend(str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            cli.append(str(value))
        else:
            raise ConfigError(
                f"argument {raw_name!r} has unsupported value type "
                f"{type(value).__name__}"
            )
    return cli

# END arguments_to_cli

# BEGIN effective_training_yaml — upstream lines 253-285
def effective_training_yaml(
    source: str,
    parsed_args: Mapping[str, Any],
    argument_names: Sequence[str],
    env: Mapping[str, str],
) -> str:
    """Merge typed trainer options and actual torchrun settings into the snapshot."""
    config = yaml.safe_load(source)
    launcher = config["launcher"]
    arguments = config.setdefault("arguments", {})
    launcher_options = {
        "target_model_path": "target_model",
        "train_data_path": "train_data",
        "train_hidden_states_path": "train_hidden_states",
        "output_dir": "output_dir",
        "teacher_draft_path": "teacher_draft",
    }
    for name in dict.fromkeys([*arguments, *argument_names]):
        value = parsed_args[name]
        if name in launcher_options:
            launcher[launcher_options[name]] = value
            arguments.pop(name, None)
        else:
            arguments[name] = value
    # Paths can also have been supplied through launcher fields, rather than
    # arguments. Record the parsed values to avoid stale duplicate settings.
    for name, key in launcher_options.items():
        if name in parsed_args and (parsed_args[name] is not None or key in launcher):
            launcher[key] = parsed_args[name]
    for key in ("nnodes", "node_rank", "nproc_per_node", "master_addr", "master_port"):
        value = env.get(f"PET_{key.upper()}", env[key.upper()])
        launcher[key] = value if key == "master_addr" else int(value)
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)

# END effective_training_yaml
