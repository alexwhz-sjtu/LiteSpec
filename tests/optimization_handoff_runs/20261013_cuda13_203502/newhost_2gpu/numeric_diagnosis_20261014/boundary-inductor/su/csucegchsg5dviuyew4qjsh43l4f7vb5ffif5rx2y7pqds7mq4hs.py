# AOT ID: ['2_backward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/boundary-inductor/3u/c3ugapbuary2lg7nnql7fvtgthnjicqb77i6bwmtsnaxgumpevxx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.expand, aten.mul, aten.neg, aten.unsqueeze, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
# Graph fragment:
#   %expand : Tensor "f32[s15][0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [%primals_4]), kwargs = {})
#   %mul_6 : Tensor "f32[s15][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_6), kwargs = {})
#   %neg_1 : Tensor "f32[s15][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_6,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[s15, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%neg_1, 1), kwargs = {})
#   %full_default : Tensor "f32[s47, s84][s84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%primals_1, %primals_2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : Tensor "f32[s47, s84][s84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, -1, %unsqueeze, %unsqueeze_1), kwargs = {})
#   return %scatter_add
triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_0 = async_compile.triton('triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/boundary-inductor/uk/cuk4tbrnxmztlezmdrx2tlqdgdwqzyvkevrdirbbqf4u2szfse2i.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.expand, aten.mul, aten.neg, aten.unsqueeze, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
# Graph fragment:
#   %unsqueeze : Tensor "i64[s15, 1][1, 1]cuda:0" = PlaceHolder[target=unsqueeze]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_6 : Tensor "f32[s15][1]cuda:0" = PlaceHolder[target=primals_6]
#   %scatter_add : Tensor "f32[s47, s84][s84, 1]cuda:0" = PlaceHolder[target=scatter_add]
#   %expand : Tensor "f32[s15][0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [%primals_4]), kwargs = {})
#   %mul_6 : Tensor "f32[s15][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_6), kwargs = {})
#   %neg_1 : Tensor "f32[s15][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_6,), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[s15, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%neg_1, 1), kwargs = {})
#   %full_default : Tensor "f32[s47, s84][s84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%primals_1, %primals_2], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : Tensor "f32[s47, s84][s84, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, -1, %unsqueeze, %unsqueeze_1), kwargs = {})
#   return %buf1
triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_1 = async_compile.triton('triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp0 < ks0")
    tmp5 = tmp3 * tmp4
    tmp6 = -tmp5
    tl.atomic_add(out_ptr0 + (tmp0 + ks0*x0), tmp6, xmask, sem='relaxed')
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
        primals_1, primals_2, primals_4, primals_6, unsqueeze, tangents_1 = args
        args.clear()
        s47 = primals_1
        s84 = primals_2
        s15 = primals_4
        assert_size_stride(primals_6, (s15, ), (1, ))
        assert_size_stride(unsqueeze, (s15, 1), (1, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s47, s84), (s84, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.expand, aten.mul, aten.neg, aten.unsqueeze, aten.new_zeros, aten.scatter_add]
            triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_0_xnumel = s47*s84
            stream0 = get_raw_stream(0)
            triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_0.run(buf0, triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_0_xnumel, stream=stream0)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.expand, aten.mul, aten.neg, aten.unsqueeze, aten.new_zeros, aten.scatter_add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_unsqueeze_1.run(unsqueeze, tangents_1, primals_6, buf0, s84, s15, stream=stream0)
            del primals_6
            del tangents_1
            del unsqueeze
        return (None, None, buf0, None, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 32
    primals_2 = 151936
    primals_4 = 32
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    unsqueeze = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_4, primals_6, unsqueeze, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
