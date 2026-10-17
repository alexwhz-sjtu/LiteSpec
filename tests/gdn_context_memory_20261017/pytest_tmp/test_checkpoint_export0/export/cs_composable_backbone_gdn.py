# Generated inference bundle from composable/backbone/gdn.py

"""GDN backbone with independent per-layer slot embeddings and prefix memory."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from .cs_dflash_kernels import DEFAULT_DFLASH_KERNELS
from .cs_composable_contracts import BackboneOutput
from .cs_composable_backbone_attention_gated_deltanet import GDNContextMemoryMixer
from .cs_composable_backbone_base import Backbone
from .cs_composable_backbone_gated_delta_state import GDNBackboneState, GDNLayerState
from .cs_composable_backbone_gdn_events import build_event_layout

class GDNLayer(nn.Module):

    def __init__(self, config, recipe):
        super().__init__()
        self.block_size = config.block_size
        self.position_embedding = nn.Embedding(config.block_size, config.hidden_size)
        kernels = DEFAULT_DFLASH_KERNELS
        self.input_norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.mixer = GDNContextMemoryMixer(config, recipe)
        self.mlp = kernels.make_mlp(config)

    def forward(self, hidden, context, layout, previous=None):
        n, length, d = hidden.shape
        hidden = (hidden.reshape(n, -1, self.block_size, d) + self.position_embedding.weight).reshape(n, length, d)
        mixed, state = self.mixer(self.input_norm(hidden), context, layout, self.block_size, previous)
        hidden = hidden + mixed
        return (hidden + self.mlp(self.post_attention_norm(hidden)), state)

class GDNBackbone(Backbone):

    def __init__(self, config):
        super().__init__()
        recipe = config.composable_config['backbone']
        self.block_size = config.block_size
        self.layers = nn.ModuleList([GDNLayer(config, recipe) for _ in recipe['layers']])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.last_event_count = 0

    def prepare_cache(self, inputs):
        """Validate the complete commit before publishing any layer's new state."""
        if inputs.draft_state is None:
            return (None, None)
        if inputs.draft_state.layers:
            raise ValueError('GDN cannot reuse an attention KV cache')
        old = inputs.draft_state.backbone_state
        n, a = inputs.anchor_positions.shape
        if a != 1 or not inputs.block_keep_mask.all():
            raise ValueError('Cached GDN inference requires one active block per sample')
        if old is not None and (not isinstance(old, GDNBackboneState)):
            raise ValueError('Unsupported backbone state type')
        device = inputs.draft_embeddings.device
        docs = torch.zeros(n, dtype=torch.long, device=device) if inputs.anchor_document_ids is None else inputs.anchor_document_ids[:, 0].to(device)
        if old is None:
            start = torch.zeros(n, dtype=torch.long, device=device)
            reset = torch.ones(n, dtype=torch.bool, device=device)
        else:
            if old.next_positions.shape != (n,) or old.document_ids.shape != (n,):
                raise ValueError('GDN cached batch size changed')
            reset = docs != old.document_ids.to(device)
            start = torch.where(reset, 0, old.next_positions.to(device))
        s = inputs.target_hidden_states.size(1)
        if inputs.context_position_ids is None and s and (start > 0).any():
            raise ValueError('GDN continuation requires absolute context_position_ids')
        positions = torch.arange(s).expand(n, -1) if inputs.context_position_ids is None else inputs.context_position_ids.detach().cpu()
        valid = torch.ones(n, s, dtype=torch.bool) if inputs.context_valid_mask is None else inputs.context_valid_mask.detach().cpu().bool()
        context_docs = torch.zeros(n, s, dtype=torch.long) if inputs.document_ids is None else inputs.document_ids.detach().cpu()
        end = inputs.anchor_positions[:, 0].to(device)
        for sample in range(n):
            selected = positions[sample, valid[sample]]
            expected = torch.arange(int(start[sample]), int(end[sample]))
            if int(end[sample]) < int(start[sample]) or not torch.equal(selected, expected):
                raise ValueError('GDN commit has duplicate, missing, future or out-of-order context positions')
            if (context_docs[sample, valid[sample]] != int(docs[sample])).any():
                raise ValueError('Cached GDN context must belong to the active document')
        previous = {}
        for index, layer in enumerate(self.layers):
            mixer = layer.mixer
            shape = (n, mixer.hv, mixer.dk, mixer.dv)
            history = mixer.context_kernel - 1
            tail_shape = (n, mixer.key_dim + mixer.value_dim, history)
            state = None if old is None else old.layers.get(index)
            if old is not None and state is None:
                raise ValueError('Incomplete GDN layer state')
            if state is not None:
                if state.context_conv_kernel_size != mixer.context_kernel or state.recurrent_state.shape != shape:
                    raise ValueError('GDN state layout does not match the model recipe')
                if history and (state.kv_conv_state is None or state.kv_conv_state.shape != tail_shape):
                    raise ValueError('Invalid GDN convolution state layout')
                if not history and state.kv_conv_state is not None:
                    raise ValueError('kernel=1 must not have historical convolution state')
                matrix = state.recurrent_state.to(device=device, dtype=torch.float32)
                matrix = matrix.masked_fill(reset[:, None, None, None], 0)
                tail = None if not history else state.kv_conv_state.to(device).masked_fill(reset[:, None, None], 0)
            else:
                matrix = torch.zeros(shape, device=device, dtype=torch.float32)
                tail = None if not history else inputs.draft_embeddings.new_zeros(tail_shape)
            previous[index] = GDNLayerState(matrix, tail, mixer.context_kernel)
        return (previous, GDNBackboneState(next_positions=end.detach().clone(), document_ids=docs.detach().clone()))

    def forward(self, context):
        inputs = context.inputs
        previous, updated = self.prepare_cache(inputs)
        layout = build_event_layout(inputs, self.block_size)
        self.last_event_count = layout.num_events
        hidden = inputs.draft_embeddings
        if len(context.context_hidden_states) != len(self.layers):
            raise ValueError('Context layer count does not match the GDN backbone')
        for index, (layer, c) in enumerate(zip(self.layers, context.context_hidden_states)):
            if self.gradient_checkpointing and self.training:

                def operation(x, features, module=layer):
                    return module(x, features, layout)[0]
                hidden = checkpoint(operation, hidden, c, use_reentrant=False)
            else:
                hidden, state = layer(hidden, c, layout, None if previous is None else previous[index])
                if updated is not None:
                    updated.layers[index] = state
        if updated is not None:
            inputs.draft_state.backbone_state = updated
        valid = inputs.block_keep_mask.repeat_interleave(self.block_size, -1)
        return BackboneOutput(self.norm(hidden) * valid[..., None].to(hidden.dtype), inputs.draft_state)
