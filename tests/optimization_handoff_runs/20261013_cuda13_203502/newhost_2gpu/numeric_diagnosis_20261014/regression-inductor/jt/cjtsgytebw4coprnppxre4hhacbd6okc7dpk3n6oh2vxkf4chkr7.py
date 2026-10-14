# AOT ID: ['16_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/s7/cs7hzykq57mh3xz7pxoql4hgapqyzajoej27rix7necf3afb7kaq.py
# Topologically Sorted Source Nodes: [float_1, values, mul, sum_1], Original ATen: [aten._to_copy, aten.binary_cross_entropy_with_logits, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   mul => mul_9
#   sum_1 => sum_1
#   values => abs_1, exp, full_default, log1p, minimum, mul_6, neg, sub_6, sub_7, sub_8
# Graph fragment:
#   %primals_4 : Tensor "f32[s24, s13][1, s24]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_3 : Tensor "bf16[s24, s13][1, s24]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_5 : Tensor "f32[s24, 1][1, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %convert_element_type : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %sub_6 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %primals_4), kwargs = {})
#   %mul_6 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %convert_element_type), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %convert_element_type), kwargs = {})
#   %abs_1 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type,), kwargs = {})
#   %neg : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_7 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_8 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, %sub_7), kwargs = {})
#   %mul_9 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %primals_5), kwargs = {})
#   %sum_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_9,), kwargs = {dtype: torch.float32})
#   return %sum_1
triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_0 = async_compile.triton('triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_0 = (r0_index % ks0)
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = 1.0
        tmp2 = tmp1 - tmp0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 * tmp4
        tmp6 = 0.0
        tmp7 = triton_helpers.minimum(tmp6, tmp4)
        tmp8 = tl_math.abs(tmp4)
        tmp9 = -tmp8
        tmp10 = libdevice.exp(tmp9)
        tmp11 = libdevice.log1p(tmp10)
        tmp12 = tmp7 - tmp11
        tmp13 = tmp5 - tmp12
        tmp15 = tmp13 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp17, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5 = args
        args.clear()
        s88 = primals_1
        s3 = primals_2
        primals_3_size = primals_3.size()
        s24 = primals_3_size[0]
        s13 = primals_3_size[1]
        assert_size_stride(primals_3, (s24, s13), (1, s24))
        assert_size_stride(primals_4, (s24, s13), (1, s24))
        assert_size_stride(primals_5, (s24, 1), (1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, values, mul, sum_1], Original ATen: [aten._to_copy, aten.binary_cross_entropy_with_logits, aten.mul, aten.sum]
            triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_0_r0_numel = s13*s24
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_0.run(primals_4, primals_3, primals_5, buf0, s24, 1, triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_0_r0_numel, stream=stream0)
        return (buf0, primals_3, primals_4, primals_5, s24, s13, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 7
    primals_2 = 513
    primals_3 = rand_strided((7, 513), (1, 7), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((7, 513), (1, 7), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((7, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
