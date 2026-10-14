#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "/root/.cache/uv/environments-v2/litespec-cp3.11-befba8bee4099476/lib/python3.11/site-packages/sglang/jit_kernel/csrc/elementwise/qknorm.cuh"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qknorm, (QKNormKernel<128, true, bf16_t>::run));
