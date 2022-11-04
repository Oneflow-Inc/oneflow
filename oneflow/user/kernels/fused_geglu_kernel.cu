/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {

template<typename T>
__device__ T Gelu(T x) {
  return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + erf(static_cast<T>(M_SQRT1_2) * x));
}

/*! Noted
 * mixture precision isn't allowed
 * hence we need to explicitly cast half or nv_bfloat16 type to float
 * and after finishing the gelu calculation, explicitly cast back
 */
template<>
__device__ half Gelu(half x) {
  return static_cast<half>(Gelu<float>(static_cast<float>(x)));
}
template<>
__device__ nv_bfloat16 Gelu(nv_bfloat16 x) {
  return static_cast<nv_bfloat16>(Gelu<float>(static_cast<float>(x)));
}

template<typename T>
__global__ void FusedGegluForwardGpu(const int in_size, const int out_size, const int num_sample,
                                     const T* matmul, const T* b, T* y) {
  // obtain the index of current thread
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size * num_sample) { return; }

  // obtain index of row and col in the output tensor
  const int64_t out_row = i / out_size;
  const int64_t out_col = i - out_row * out_size;

  // obtain col index in both matmul tensor and bias tensor
  const int64_t x1_col = out_col;
  const int64_t x2_col = out_col + out_size;

  // obtain element before gelu and element-wise product
  T hidden_state = matmul[2 * out_row * out_size + x1_col] + b[x1_col];
  T gate = matmul[2 * out_row * out_size + x2_col] + b[x2_col];

  // calculate gelu
  T gelu_gate = Gelu<T>(gate);

  // calculate element-wise product
  y[i] = gelu_gate * hidden_state;
}

}  // namespace

template<typename T>

class GpuFusedGegluKernel final : public user_op::OpKernel {
 public:
  GpuFusedGegluKernel() = default;
  ~GpuFusedGegluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    // obtain corresponding tensors from the context
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* w = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* matmul_out = ctx->Tensor4ArgNameAndIndex("matmul_out", 0);

    // obtain dimensions
    const int64_t in_size = in->shape_view().At(1);
    const int64_t out_size = out->shape_view().At(1);
    const int64_t num_samples = in->shape_view().At(0);

    // calculate X*W through cuBLAS
    // ref -> reduce_kernel.cpp -> matmul
    auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
        DeviceType::kCUDA, in->data_type(), ep::primitive::BlasTransposeType::N,
        ep::primitive::BlasTransposeType::T);
    CHECK(matmul);
    /* Launch(Stream* stream, size_t m, size_t n, size_t k, Scalar alpha, const void* a,
                  const void* b, Scalar beta, void* c) = 0; */
    matmul->Launch(ctx->stream(), num_samples, out_size * 2, in_size, 1.0, in->dptr(), w->dptr(),
                   0.0, matmul_out->mut_dptr());

    // invoke fused geglu kernel
    RUN_CUDA_KERNEL((FusedGegluForwardGpu<T>), ctx->stream(),
                    out_size * num_samples, /* number of threads */
                    in_size,                /* in_size */
                    out_size,               /* out_size */
                    num_samples,            /* element_cnt */
                    matmul_out->dptr<T>(),  /* matmul result */
                    b->dptr<T>(),           /* bias */
                    out->mut_dptr<T>()      /* output tensor */
    );
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_FUSED_GEGLU_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("fused_geglu")                                  \
      .SetCreateFn<GpuFusedGegluKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_FUSED_GEGLU_KERNEL(float)
REGISTER_GPU_FUSED_GEGLU_KERNEL(double)
REGISTER_GPU_FUSED_GEGLU_KERNEL(half)
REGISTER_GPU_FUSED_GEGLU_KERNEL(nv_bfloat16)

}  // namespace oneflow