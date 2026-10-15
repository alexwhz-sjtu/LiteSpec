# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: author_repository
# Repository: z-lab/dflash
# Revision: 07ebd93db9f472af339b644bb70221ad8428328a
# Source: dflash/model.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/z-lab__dflash/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py
# License / upstream notices: refrence/licenses/dflash-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Inference reference shifts along its sequence; multi-anchor training must preserve framework block isolation.


# BEGIN _grouped_dynamic_convolve — upstream lines 478-489
def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)

# END _grouped_dynamic_convolve

# BEGIN GroupedDynamicCausalConv — upstream lines 492-512
class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size)

# END GroupedDynamicCausalConv
