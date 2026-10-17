# Generated inference bundle from composable/modeling.py

"""Thin deployable orchestration of context, backbone and proposal."""
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
from .cs_dflash import build_target_layer_ids
from .cs_dflash2 import DFlashGroupedConv
from .cs_registry import register_draft
from .cs_composable_backbone import build_backbone
from .cs_composable_configuration import ComposableDraftConfig
from .cs_composable_context import build_context
from .cs_composable_context_hidden_gated import HiddenGatedContext
from .cs_composable_proposal import build_proposal

class ComposableDraftModel(Qwen3PreTrainedModel):
    config_class = ComposableDraftConfig
    method_config_key = 'composable_config'
    _no_split_modules = ['DraftLayer']
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        config.validate_recipe()
        recipe = config.composable_config
        if recipe['context'].get('target_layer_ids') is None:
            recipe['context']['target_layer_ids'] = list(range(config.num_target_layers)) if config.num_target_layers < 4 else sorted(set(build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)))
        config.validate_recipe()
        self.target_layer_ids = recipe['context']['target_layer_ids']
        self.block_size = config.block_size
        self.mask_token_id = recipe.get('mask_token_id')
        self.context = build_context(config)
        self.backbone = build_backbone(config)
        self.proposal = build_proposal(config)
        self.post_init()
        for module in self.modules():
            if isinstance(module, DFlashGroupedConv):
                torch.nn.init.zeros_(module.kernel_projection.weight)
            if isinstance(module, HiddenGatedContext):
                module.reset_fusion_prior()
        head = getattr(self.proposal, 'markov_head', None)
        if head is not None:
            torch.nn.init.zeros_(head.markov_w2.weight)
        selector = getattr(self.proposal, 'candidate_selector', None)
        if selector is not None:
            torch.nn.init.zeros_(selector.successor_codebook)

    def forward(self, inputs, *, shared_ops=None):
        if inputs.mode not in ('features', 'teacher_forcing', 'sample'):
            raise ValueError('mode must be features, teacher_forcing, or sample')
        if inputs.temperature < 0:
            raise ValueError('temperature must be >= 0')
        if self.training and inputs.draft_state is not None:
            raise ValueError('Persistent inference state is not accepted in training')
        if inputs.draft_state is not None and inputs.anchor_positions.size(1) != 1:
            raise ValueError('Cached inference expects one current block per sample')
        context = self.context(inputs)
        features = self.backbone(context)
        output = self.proposal(features, context, shared_ops)
        output.backbone_hidden_states = features.draft_hidden_states
        return output

# Direct closure for Transformers 4.57 local-directory loading.
from .cs_composable_configuration import __name__ as _export_cs_composable_configuration
from .cs_composable_proposal import __name__ as _export_cs_composable_proposal
from .cs_composable_proposal_sequential_rnn import __name__ as _export_cs_composable_proposal_sequential_rnn
from .cs_composable_proposal_parallel import __name__ as _export_cs_composable_proposal_parallel
from .cs_composable_proposal_base import __name__ as _export_cs_composable_proposal_base
from .cs_composable_contracts import __name__ as _export_cs_composable_contracts
from .cs_dlite import __name__ as _export_cs_dlite
from .cs_sampling import __name__ as _export_cs_sampling
from .cs_registry import __name__ as _export_cs_registry
from .cs_flex_attention import __name__ as _export_cs_flex_attention
from .cs_torch_compat import __name__ as _export_cs_torch_compat
from .cs_composable_proposal_selector import __name__ as _export_cs_composable_proposal_selector
from .cs_dflash2 import __name__ as _export_cs_dflash2
from .cs_dflash_kernels import __name__ as _export_cs_dflash_kernels
from .cs_dflash import __name__ as _export_cs_dflash
from .cs_composable_proposal_rnn import __name__ as _export_cs_composable_proposal_rnn
from .cs_composable_proposal_markov import __name__ as _export_cs_composable_proposal_markov
from .cs_dspark import __name__ as _export_cs_dspark
from .cs_composable_context_hidden_gated import __name__ as _export_cs_composable_context_hidden_gated
from .cs_composable_context_base import __name__ as _export_cs_composable_context_base
from .cs_composable_context import __name__ as _export_cs_composable_context
from .cs_composable_context_hidden_mlp import __name__ as _export_cs_composable_context_hidden_mlp
from .cs_composable_backbone import __name__ as _export_cs_composable_backbone
from .cs_composable_backbone_transformer import __name__ as _export_cs_composable_backbone_transformer
from .cs_composable_backbone_layer import __name__ as _export_cs_composable_backbone_layer
from .cs_composable_backbone_operators import __name__ as _export_cs_composable_backbone_operators
from .cs_composable_backbone_operators_dflash2_conv import __name__ as _export_cs_composable_backbone_operators_dflash2_conv
from .cs_composable_backbone_attention import __name__ as _export_cs_composable_backbone_attention
from .cs_composable_backbone_attention_mla import __name__ as _export_cs_composable_backbone_attention_mla
from .cs_composable_backbone_attention_base import __name__ as _export_cs_composable_backbone_attention_base
from .cs_composable_backbone_attention_access import __name__ as _export_cs_composable_backbone_attention_access
from .cs_composable_backbone_attention_access_sliding_window import __name__ as _export_cs_composable_backbone_attention_access_sliding_window
from .cs_composable_backbone_attention_access_base import __name__ as _export_cs_composable_backbone_attention_access_base
from .cs_composable_backbone_attention_access_dense import __name__ as _export_cs_composable_backbone_attention_access_dense
from .cs_composable_backbone_attention_mha import __name__ as _export_cs_composable_backbone_attention_mha
from .cs_composable_backbone_base import __name__ as _export_cs_composable_backbone_base
