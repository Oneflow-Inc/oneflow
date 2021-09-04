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
#include "oneflow/core/framework/framework.h"
#include <cuda.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename T>
__global__ void ReluForwardGpu(int64_t n, const T* in, T* out) {
  const T zero = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T in_i = in[i];
    T out_i = zero;
    if (in_i > zero) { out_i = in_i; }
    out[i] = out_i;
  }
}

template<typename T>
__global__ void ReluBackwardGpu(int64_t n, const T* y, const T* dy, T* dx) {
  const T zero = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = y[i] > zero ? dy[i] : zero; }
}

}  // namespace

class ReluNvBFloat16Kernel final : public OpKernel {
 public:
  ReluNvBFloat16Kernel() = default;
  ~ReluNvBFloat16Kernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t n = in->shape().elem_cnt();
    ReluForwardGpu<nv_bfloat16>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, reinterpret_cast<const nv_bfloat16*>(in->dptr()),
            reinterpret_cast<nv_bfloat16*>(out->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ReluGradNvBFloat16Kernel final : public OpKernel {
 public:
  ReluGradNvBFloat16Kernel() = default;
  ~ReluGradNvBFloat16Kernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t n = y->shape().elem_cnt();
    ReluBackwardGpu<nv_bfloat16>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, reinterpret_cast<const nv_bfloat16*>(y->dptr()),
            reinterpret_cast<const nv_bfloat16*>(dy->dptr()),
            reinterpret_cast<nv_bfloat16*>(dx->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("relu")
    .SetCreateFn<ReluNvBFloat16Kernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("out", 0) == DataType::kBFloat16))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("relu_grad")
    .SetCreateFn<ReluGradNvBFloat16Kernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("dx", 0) == DataType::kBFloat16))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));
      return Maybe<void>::Ok();
    });

}  // namespace user_op

}  // namespace oneflow

#endif  // defined(CUDA_VERSION) && CUDA_VERSION >= 11000
