# Generated inference bundle from composable/configuration.py

"""Serializable model recipe; target dimensions follow the normal HF config."""
from copy import deepcopy
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

class ComposableDraftConfig(Qwen3Config):
    model_type = 'litespec_composable'

    def __init__(self, composable_config=None, block_size=8, num_target_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.block_size = int(block_size)
        self.num_target_layers = int(num_target_layers or self.num_hidden_layers)
        self.composable_config = deepcopy(composable_config or {'context': {'type': 'dflash'}, 'backbone': {'type': 'transformer', 'layers': [{'attention': {'representation': 'mha', 'access': 'dense', 'backend': 'sdpa'}} for _ in range(self.num_hidden_layers)]}, 'proposal': {'type': 'parallel'}, 'objective': {'type': 'hard_ce'}})
        self.architectures = ['ComposableDraftModel']
        self.schema_version = 'composable_v1'
        self.validate_recipe()

    def validate_recipe(self):
        recipe = self.composable_config
        if self.block_size < 2:
            raise ValueError('block_size includes one anchor and must be >= 2')
        if recipe['context']['type'] not in ('dflash', 'dflare'):
            raise ValueError('Only dflash/dflare hidden injection is supported; target KV is deferred')
        layers = recipe['backbone'].get('layers')
        if recipe['backbone'].get('type', 'transformer') != 'transformer' or not layers:
            raise ValueError('backbone.layers must explicitly describe each transformer layer')
        if len(layers) != self.num_hidden_layers:
            raise ValueError('backbone.layers length must equal num_hidden_layers')
        for layer in layers:
            attn = layer['attention']
            if attn['representation'] not in ('mha', 'mla'):
                raise ValueError('Only mha and mla are supported')
            access = attn.get('access', 'dense')
            if access not in ('dense', 'sliding_window'):
                raise ValueError('Only dense and sliding_window access are supported')
            if access == 'sliding_window' and int(attn.get('window_size', 0)) <= 0:
                raise ValueError('sliding_window requires a positive window_size')
            if attn.get('backend', 'sdpa') not in ('sdpa', 'eager', 'flex_attention'):
                raise ValueError('Unsupported attention backend')
            if attn['representation'] == 'mla':
                for key in ('kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim'):
                    if int(attn.get(key, 0)) <= 0:
                        raise ValueError('MLA requires a positive ' + key)
                if attn['qk_rope_head_dim'] % 2:
                    raise ValueError('MLA rotary dimension must be even')
            if attn.get('backend') == 'flex_attention':
                key_dim = attn['qk_nope_head_dim'] + attn['qk_rope_head_dim'] if attn['representation'] == 'mla' else self.head_dim
                value_dim = attn['v_head_dim'] if attn['representation'] == 'mla' else self.head_dim
                if min(key_dim, value_dim) < 16:
                    raise ValueError('FlexAttention requires key/value head dimensions >= 16')
                if attn.get('flex_backend', 'TRITON') not in ('TRITON', 'FLASH'):
                    raise ValueError('Unsupported FlexAttention kernel backend')
            operator = layer.get('local_operator', {'type': 'none'})
            if operator['type'] not in ('none', 'dflash2_conv'):
                raise ValueError('Unsupported local operator')
        proposal = recipe['proposal']['type']
        if proposal not in ('parallel', 'markov', 'rnn', 'sequential_rnn', 'selector'):
            raise ValueError('Unsupported proposal')
        objective = recipe['objective']['type']
        if objective not in ('hard_ce', 'soft_ce', 'forward_kl', 'tv', 'nla', 'lk_alpha', 'dpace', 'bv'):
            raise ValueError('Unsupported objective')
        if objective == 'bv' and proposal == 'selector':
            raise ValueError('BV + selector is not supported')
        ids = recipe['context'].get('target_layer_ids')
        if ids is not None and (not ids or ids != sorted(set(ids)) or ids[0] < 0 or (ids[-1] >= self.num_target_layers)):
            raise ValueError('target_layer_ids must be unique, sorted, and within the target')
