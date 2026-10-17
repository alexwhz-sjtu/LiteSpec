# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Adapted from Transformers Qwen3.5, aad13b87ed59f2afcfaebc985f403301887a35fc.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""GDN context writes and identity-transition block reads (not softmax attention).

Q/Z come from draft blocks; K/V/a/b come from confirmed target context. Q and
context use independent causal depthwise convolutions. Read-only events reuse
the official chunk recurrence with exactly zero beta and log-decay.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..gated_delta_state import GDNLayerState


def l2norm(x):
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def reference_event_scan(q, k, v, g, beta, cu_seqlens_cpu, initial_state=None):
    """Small, differentiable FP32 oracle; never a silent CUDA training fallback."""
    outputs, states = [], []
    boundaries = cu_seqlens_cpu.tolist()
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (
            v.new_zeros(v.size(2), k.size(-1), v.size(-1), dtype=torch.float32)
            if initial_state is None else initial_state[seq].float()
        )
        for t in range(start, end):
            qt, kt = l2norm(q[0, t]), l2norm(k[0, t])
            repeat = v.size(2) // k.size(2)
            if repeat != 1:
                qt = qt.repeat_interleave(repeat, 0)
                kt = kt.repeat_interleave(repeat, 0)
            state = state * g[0, t].float().exp()[:, None, None]
            old = (state * kt[:, :, None]).sum(-2)
            delta = beta[0, t].float()[:, None] * (v[0, t].float() - old)
            state = state + kt[:, :, None] * delta[:, None, :]
            outputs.append((state * (qt / math.sqrt(q.size(-1)))[:, :, None]).sum(-2))
        states.append(state)
    return torch.stack(outputs)[None].to(q.dtype), torch.stack(states)


class GDNRMSNormGated(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * F.silu(z.float())).to(dtype)


class GDNContextMemoryMixer(nn.Module):
    def __init__(self, config, recipe):
        super().__init__()
        self.hk = recipe["linear_num_key_heads"]
        self.hv = recipe["linear_num_value_heads"]
        self.dk = recipe["linear_key_head_dim"]
        self.dv = recipe["linear_value_head_dim"]
        self.q_kernel = recipe["query_conv_kernel_size"]
        self.context_kernel = recipe["context_conv_kernel_size"]
        self.backend = recipe["backend"]
        self.key_dim, self.value_dim = self.hk * self.dk, self.hv * self.dv
        self.in_proj_q = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.in_proj_kv = nn.Linear(config.hidden_size, self.key_dim + self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.hv, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.hv, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.query_conv = nn.Conv1d(
            self.key_dim, self.key_dim, self.q_kernel, groups=self.key_dim, bias=False
        )
        channels = self.key_dim + self.value_dim
        self.context_conv = nn.Conv1d(
            channels, channels, self.context_kernel, groups=channels, bias=False
        )
        self.A_log = nn.Parameter(torch.empty(self.hv))
        self.dt_bias = nn.Parameter(torch.ones(self.hv))
        self.norm = GDNRMSNormGated(self.dv, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.reset_gdn_parameters()

    def reset_gdn_parameters(self):
        with torch.no_grad():
            self.dt_bias.fill_(1)
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(0, 16).clamp_min_(1e-6).log_())

    def context_convolution(self, raw, layout, previous):
        """Pack real context only; document loops do not replicate prefixes."""
        outputs, tails, offset = [], [], 0
        history = self.context_kernel - 1
        for sample, length in zip(layout.samples, layout.context_lengths):
            part = raw[offset:offset + length].T[None]
            tail = (
                part.new_zeros(1, raw.size(-1), history)
                if previous is None or previous.kv_conv_state is None
                else previous.kv_conv_state[sample:sample + 1].to(part)
            )
            joined = torch.cat((tail, part), -1)
            if length:
                value = F.silu(self.context_conv(joined)).squeeze(0).T
            else:
                value = raw[offset:offset] + self.context_conv.weight.sum() * 0
            outputs.append(value)
            if history:
                tails.append(joined[0, :, -history:])
            offset += length
        value = torch.cat(outputs) if outputs else raw + self.context_conv.weight.sum() * 0
        return value, torch.stack(tails) if tails else None

    def forward(self, hidden, context, layout, block_size, previous=None):
        n, length, _ = hidden.shape
        a = length // block_size
        raw_q = self.in_proj_q(hidden).reshape(n * a, block_size, self.key_dim).transpose(1, 2)
        q = F.silu(self.query_conv(F.pad(raw_q, (self.q_kernel - 1, 0))))
        q = q.transpose(1, 2).reshape(n * length, self.hk, self.dk)
        z = self.in_proj_z(hidden).reshape(n, length, self.hv, self.dv)
        c = context.flatten(0, 1).index_select(0, layout.context_indices)
        raw_kv = self.in_proj_kv(c)
        mixed, tails = self.context_convolution(raw_kv, layout, previous)
        k, v = mixed.split((self.key_dim, self.value_dim), -1)
        k, v = k.reshape(-1, self.hk, self.dk), v.reshape(-1, self.hv, self.dv)
        beta = self.in_proj_b(c).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(c).float() + self.dt_bias.float())
        # index_copy is differentiable, including empty source indices. Read
        # gates MUST stay zero: never activate/sigmoid these events in-kernel.
        e = layout.num_events
        if e:
            qe = q.new_zeros(1, e, self.hk, self.dk).index_copy(
                1, layout.read_events, q.index_select(0, layout.query_indices)[None]
            )
            ke = k.new_zeros(1, e, self.hk, self.dk).index_copy(1, layout.write_events, k[None])
            ve = v.new_zeros(1, e, self.hv, self.dv).index_copy(1, layout.write_events, v[None])
            ge = g.new_zeros(1, e, self.hv).index_copy(1, layout.write_events, g[None])
            be = beta.new_zeros(1, e, self.hv).index_copy(1, layout.write_events, beta[None])
            initial = None if previous is None else previous.recurrent_state
            if self.backend == "reference" or (self.backend == "auto" and not hidden.is_cuda):
                output, final = reference_event_scan(qe, ke, ve, ge, be, layout.cu_seqlens_cpu, initial)
            else:
                if not hidden.is_cuda:
                    raise ValueError("GDN FLA backend requires CUDA; use backend=reference on CPU")
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule

                output, final = chunk_gated_delta_rule(
                    qe, ke, ve, ge, be,
                    initial_state=initial, output_final_state=previous is not None,
                    use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=False,
                    use_gate_in_kernel=False, state_v_first=False,
                    cu_seqlens=layout.cu_seqlens, cu_seqlens_cpu=layout.cu_seqlens_cpu,
                )
            selected = output[0].index_select(0, layout.read_events)
            out = v.new_zeros(n * length, self.hv, self.dv).index_copy(0, layout.query_indices, selected)
        else:
            out = z.flatten(0, 1) * 0 + (q.sum() + mixed.sum() + beta.sum() + g.sum()) * 0
            final = None
        out = self.norm(out.reshape(n, length, self.hv, self.dv), z)
        result = self.out_proj(out.flatten(-2))
        state = None
        if previous is not None:
            state = GDNLayerState(final.float().detach(), None if tails is None else tails.detach(), self.context_kernel)
        return result, state
