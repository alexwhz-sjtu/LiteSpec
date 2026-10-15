# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: existing_framework
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/modeling/draft/dflash2.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/modeling/draft/dflash2.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Block-local training layout and identity initialization reference.


# BEGIN DFlashGroupedConv — upstream lines 29-139
class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution within each proposal block.

    One projection of the sublayer input produces the dynamic kernel deltas
    for both the input and output side.  ``base_kernel`` starts as an identity,
    which makes enabling the module a stable extension of a DFlash backbone.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if taps < 1:
            raise ValueError("DFlash2 conv_kernel_size must be >= 1")
        if taps > block_size:
            raise ValueError(
                "DFlash2 conv_kernel_size must not exceed block_size, got "
                f"conv_kernel_size={taps}, block_size={block_size}"
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}"
            )

        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size

        # [input/output side, tap, channel], matching SGLang's loader contract.
        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.taps * self.num_groups,
            bias=False,
        )
        nn.init.zeros_(self.kernel_projection.weight)

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape
        if seq_len % self.block_size:
            raise ValueError(
                "DFlash2 convolution sequence length must be divisible by "
                f"block_size={self.block_size}, got {seq_len}"
            )

        num_blocks = seq_len // self.block_size
        blocks = hidden_states.reshape(
            bsz,
            num_blocks,
            self.block_size,
            self.num_groups,
            self.group_size,
        )
        dynamic = delta.reshape(
            bsz,
            num_blocks,
            self.block_size,
            self.taps,
            self.num_groups,
        )
        base = self.base_kernel[side].reshape(
            1,
            1,
            1,
            self.taps,
            self.num_groups,
            self.group_size,
        )
        coefficients = base + dynamic.unsqueeze(-1)

        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(
                blocks[:, :, : self.block_size - tap],
                (0, 0, 0, 0, tap, 0),
            )
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(bsz, seq_len, hidden_size)

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.taps,
            self.num_groups,
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)

# END DFlashGroupedConv

# BEGIN Qwen3DFlash2DecoderLayer — upstream lines 142-197
class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    """Qwen3 DFlash layer with DFlash2 convolutional sublayer wrappers."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
        *,
        attention_conv: DFlashGroupedConv,
        mlp_conv: DFlashGroupedConv,
    ) -> None:
        super().__init__(config, layer_idx, kernels)
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(
            hidden_states,
            attention_kernel,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return residual + hidden_states

# END Qwen3DFlash2DecoderLayer
