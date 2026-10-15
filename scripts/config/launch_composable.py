"""Launch composable training through the existing YAML/torchrun infrastructure."""

import argparse
from pathlib import Path

try:
    from scripts.config.training_launcher import (
        arguments_to_cli,
        launch_training,
        launcher_environment,
        load_yaml_config,
    )
except ModuleNotFoundError:
    from training_launcher import (
        arguments_to_cli,
        launch_training,
        launcher_environment,
        load_yaml_config,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("composable.yaml")
    )
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    launcher, arguments, snapshot = load_yaml_config(
        args.config, identity_key="algorithm", identity_value="composable"
    )
    env = launcher_environment(launcher, yaml_snapshot=snapshot)
    if args.dry_run:
        env["DRY_RUN"] = "1"
    return launch_training(
        label="composable",
        module="scripts.composable.train",
        configured_env=env,
        configured_cli=arguments_to_cli(arguments),
        passthrough=passthrough,
        project_dir=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    raise SystemExit(main())
