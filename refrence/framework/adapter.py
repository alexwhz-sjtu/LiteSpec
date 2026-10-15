# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: framework_integration_baseline
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/training/algorithms.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/training/algorithms.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Explicit adapter whitelist must be extended later; this task does not modify it.


# BEGIN TrainingAdapter — upstream lines 119-194
@dataclass(frozen=True)
class TrainingAdapter:
    algorithm: str
    architecture: str
    modes: tuple[str, ...] = ()

    @property
    def model_class(self):
        from litespec.modeling.draft.registry import resolve_draft

        return resolve_draft(self.architecture)

    def parse_args(self, mode=None, argv=None):
        if self.modes:
            if mode not in self.modes:
                raise ValueError(f"{self.algorithm} requires mode in {self.modes}")
        elif mode is not None:
            raise ValueError(f"{self.algorithm} does not have training modes")
        module = import_module(f"litespec.training.recipes.{self.algorithm}")
        return module.parse_args(mode=mode, argv=argv)

    def build_config(self, args, *, source_config=None, model_role=None):
        raise NotImplementedError

    def build_model(self, args, *, source_config=None, model_role=None):
        import torch

        config = self.build_config(
            args, source_config=source_config, model_role=model_role
        )
        return self.model_class(config).cuda().to(torch.bfloat16)

    def load_model(self, path, attention_backend):
        import torch

        return self.model_class.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation=attention_backend
        ).cuda()

    def prepare_batch(self, draft_online, *args, **kwargs):
        return draft_online.prepare_batch(*args, **kwargs)

    def training_step(self, draft_online, batch):
        return draft_online(**batch)

    def required_target_layers(self, model):
        return set(model.target_layer_ids)

    def optimizer_parameters(self, session):
        return None

    def enter_phase(self, session, phase):
        session.draft.train()

    def detail_metrics(self, session):
        return {}

    def log_metrics(self, session, reduced):
        return reduced

    def training_metadata(self, session):
        return {}

    def checkpoint_metadata(self, model):
        return {
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "block_size": int(model.block_size),
            "num_draft_layers": int(model.config.num_hidden_layers),
            "target_layer_ids": list(model.target_layer_ids),
            "num_target_layers": int(model.config.num_target_layers),
            "target_hidden_size": int(model.config.hidden_size),
            model.method_config_key: dict(
                getattr(model.config, model.method_config_key)
            ),
        }

# END TrainingAdapter

# BEGIN get_training_adapter — upstream lines 197-200
def get_training_adapter(algorithm):
    if algorithm not in ("dflash", "dflash2", "dspark", "dlite"):
        raise ValueError(f"Unknown training algorithm {algorithm!r}")
    return import_module(f"litespec.training.recipes.{algorithm}").ADAPTER

# END get_training_adapter
