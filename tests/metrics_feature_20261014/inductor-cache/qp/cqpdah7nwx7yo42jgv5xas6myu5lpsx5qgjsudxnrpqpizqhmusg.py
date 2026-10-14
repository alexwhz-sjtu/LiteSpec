# AOT ID: ['15_backward']
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


# kernel path: tests/metrics_feature_20261014/inductor-cache/py/cpy2w2ociqjignvfh4qmwr2vjeskdxwc5pmlt77ztnucabveie2w.py
# Topologically Sorted Source Nodes: [float_1], Original ATen: [aten.expand, aten.mul, aten._to_copy, aten.sigmoid, aten.sub]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
# Graph fragment:
#   %primals_3 : Tensor "bf16[s24, s13][1, s24]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_4 : Tensor "f32[s24, s13][1, s24]cuda:0" = PlaceHolder[target=primals_4]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_5 : Tensor "f32[s24, 1][1, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %expand : Tensor "f32[s24, s13][0, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [%primals_1, %primals_2]), kwargs = {})
#   %mul_12 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_5), kwargs = {})
#   %convert_element_type : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type,), kwargs = {})
#   %sub_13 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid, %primals_4), kwargs = {})
#   %mul_13 : Tensor "f32[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %mul_12), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[s24, s13][1, s24]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_expand_mul_sigmoid_sub_0 = async_compile.triton('triton_poi_fused__to_copy_expand_mul_sigmoid_sub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_expand_mul_sigmoid_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_expand_mul_sigmoid_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp4 = tmp2 - tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp4 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, tangents_1 = args
        args.clear()
        s24 = primals_1
        s13 = primals_2
        assert_size_stride(primals_3, (s24, s13), (1, s24))
        assert_size_stride(primals_4, (s24, s13), (1, s24))
        assert_size_stride(primals_5, (s24, 1), (1, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s24, s13), (1, s24), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1], Original ATen: [aten.expand, aten.mul, aten._to_copy, aten.sigmoid, aten.sub]
            triton_poi_fused__to_copy_expand_mul_sigmoid_sub_0_xnumel = s13*s24
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_expand_mul_sigmoid_sub_0.run(primals_3, primals_4, tangents_1, primals_5, buf0, s24, triton_poi_fused__to_copy_expand_mul_sigmoid_sub_0_xnumel, stream=stream0)
            del primals_3
            del primals_4
            del primals_5
            del tangents_1
        return (None, None, buf0, None, None, )

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
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
