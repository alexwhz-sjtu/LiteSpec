# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: framework_integration_baseline
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/modeling/draft/registry.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/modeling/draft/registry.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Current dependency collector only accepts flat sibling imports; composable needs its own export path.

# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# BEGIN draft_source_files — upstream lines 45-68
def draft_source_files(architecture: str) -> tuple[str, ...]:
    """Return the complete, flat remote-code dependency closure."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent
    pending = [resolve_draft(architecture).__module__.rsplit(".", 1)[-1] + ".py"]
    seen = set()
    while pending:
        filename = pending.pop()
        if filename in seen:
            continue
        source = root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing draft export dependency: {source}")
        seen.add(filename)
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or "." in node.module:
                    raise ValueError(
                        f"Draft exports require flat relative imports: {filename}"
                    )
                pending.append(node.module + ".py")
    return tuple(sorted(seen))

# END draft_source_files

# BEGIN register_draft — upstream lines 71-94
def register_draft(cls: Optional[type] = None, *, name: Optional[str] = None):
    """Class decorator registering a draft architecture.

    The key defaults to the class name — exactly what draft-config JSONs carry
    in ``architectures``. The class must declare ``config_class`` (the HF
    ``PretrainedConfig`` subclass it is built from) so the auto loaders can
    resolve both the model and its config from the one registry entry.
    """

    def _register(cls: type) -> type:
        key = name or cls.__name__
        if getattr(cls, "config_class", None) is None:
            raise TypeError(
                f"@register_draft: {cls.__name__} must declare config_class"
            )
        existing = DRAFT_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"draft architecture {key!r} already registered to {existing.__name__}"
            )
        DRAFT_REGISTRY[key] = cls
        return cls

    return _register(cls) if cls is not None else _register

# END register_draft
