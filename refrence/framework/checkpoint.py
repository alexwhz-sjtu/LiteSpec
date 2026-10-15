# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: framework_integration_baseline
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/training/checkpoint.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/training/checkpoint.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# draft_model.* parameter extraction and old flat export path; preserve legacy behavior.


# BEGIN export_draft — upstream lines 318-344
def export_draft(model, save_dir, *, state_dict=None):
    """Export a self-contained model without importing the training package."""
    model.config.auto_map = {
        "AutoModel": f"{type(model).__module__.rsplit('.', 1)[-1]}.{type(model).__name__}"
    }
    model.config.architectures = [type(model).__name__]
    source_root = Path(__file__).resolve().parents[1] / "modeling" / "draft"
    sources = draft_source_files(type(model).__name__)
    model.save_pretrained(save_dir, state_dict=state_dict)
    for filename in sources:
        shutil.copyfile(source_root / filename, Path(save_dir) / filename)
    # Transformers 4.57 copies only direct imports into the dynamic cache for
    # local directories. Declare the complete closure in the exported entry so
    # transitive helpers are copied too; repository sources stay unmodified.
    entry = type(model).__module__.rsplit(".", 1)[-1] + ".py"
    entry_path = Path(save_dir) / entry
    dependencies = "\n".join(
        f"from .{Path(filename).stem} import __name__ as _export_{Path(filename).stem}"
        for filename in sources
        if filename != entry
    )
    with entry_path.open("a") as stream:
        stream.write(
            "\n# Complete dependency list for the remote-code loader.\n"
            + dependencies
            + "\n"
        )

# END export_draft

# BEGIN save_checkpoint — upstream lines 347-396
def save_checkpoint(
    *,
    output_dir: str,
    name: str,
    fsdp_model: FSDP,
    draft_model,
    optimizer,
    metadata: dict,
    training_args,
    training_yaml: str | None = None,
    process_group=None,
    coordinator_global_rank: int = 0,
) -> str:
    save_dir = os.path.join(output_dir, name)
    group_rank = dist.get_rank(process_group)
    if group_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier(group=process_group)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        full_state = fsdp_model.state_dict()
        draft_state = {
            key.split("draft_model.", 1)[1]: value
            for key, value in full_state.items()
            if "draft_model." in key
        }
        algorithm = draft_model.method_config_key.removesuffix("_config")
        model_metadata = get_training_adapter(algorithm).checkpoint_metadata(
            draft_model
        )
        model_metadata.update(training_identity(training_args))
        save_distributed_training_state(
            save_dir,
            {
                **metadata,
                **model_metadata,
                **optimizer.state_dict(),
                "rng_state": capture_rng_state(),
            },
            process_group=process_group,
        )
        if group_rank == 0:
            export_draft(draft_model, save_dir, state_dict=draft_state)
            if training_yaml is not None:
                Path(save_dir, "training_config.yaml").write_bytes(
                    training_yaml.encode("utf-8")
                )
    dist.barrier(group=process_group)
    if dist.get_rank() == coordinator_global_rank:
        print(f"Saved checkpoint to {save_dir}", flush=True)
    return save_dir

# END save_checkpoint
