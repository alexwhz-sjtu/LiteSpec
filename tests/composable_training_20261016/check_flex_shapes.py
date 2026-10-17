import json
import torch
from torch import nn
from types import SimpleNamespace
from litespec.modeling.draft.composable.backbone.attention.base import AttentionForward

class Attention(AttentionForward, nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.flex_backend = "TRITON"
        self.window_size = None
        self.scaling = 16 ** -0.5
        self.attention_dropout = 0.0
        self.o_proj = nn.Identity()

    def _compute_qkv(self, draft, context, embeddings):
        q = draft.view(1, -1, 2, 16).transpose(1, 2)
        kv = torch.cat((context, draft), 1).view(1, -1, 2, 16).transpose(1, 2)
        return q, kv, kv

    def access(self, inputs, positions, valid, documents, query_length):
        q = torch.arange(query_length, device=positions.device)
        causal_context = positions[:, None, :] < q[None, :, None] + 16
        own_block = q[:, None] // 8 == q[None, :] // 8
        mask = torch.cat((causal_context, own_block[None]), -1)[:, None]
        mask[:, :, -1] = False
        return mask

torch.manual_seed(42)
records = []
eager, flex = Attention("eager").cuda(), Attention("flex_attention").cuda()
for context_length in (173, 245, 333, 77, 173):
    query_length = 256
    draft = torch.randn(1, query_length, 32, device="cuda", requires_grad=True)
    context = torch.randn(1, context_length, 32, device="cuda", requires_grad=True)
    inputs = SimpleNamespace(context_position_ids=None, context_valid_mask=None,
                             document_ids=None, draft_state=None)
    expected = eager(draft, context, None, inputs)
    expected_grads = torch.autograd.grad(expected.square().sum(), (draft, context))
    actual = flex(draft, context, None, inputs)
    actual_grads = torch.autograd.grad(actual.square().sum(), (draft, context))
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=4e-4, rtol=4e-3)
    records.append({"context_length": context_length, "max_error": (actual-expected).abs().max().item()})
    print(records[-1], flush=True)
print("FLEX_DYNAMIC_SHAPES_FORWARD_BACKWARD_PASS", flush=True)
