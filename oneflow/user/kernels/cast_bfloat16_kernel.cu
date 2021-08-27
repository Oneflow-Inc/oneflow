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

template<typename T, typename U>
__global__ void CastOnGpu(int64_t n, const T* in, U* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = static_cast<U>(in[i]); }
}

}  // namespace

class CastNvBFloat162FloatKernel final : public OpKernel {
 public:
  CastNvBFloat162FloatKernel() = default;
  ~CastNvBFloat162FloatKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t n = in->shape().elem_cnt();
    CastOnGpu<nv_bfloat16, float>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, reinterpret_cast<const nv_bfloat16*>(in->dptr()),
            reinterpret_cast<float*>(out->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class CastFloat2NvBFloat16Kernel final : public OpKernel {
 public:
  CastFloat2NvBFloat16Kernel() = default;
  ~CastFloat2NvBFloat16Kernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t n = in->shape().elem_cnt();
    CastOnGpu<float, nv_bfloat16>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, reinterpret_cast<const float*>(in->dptr()),
            reinterpret_cast<nv_bfloat16*>(out->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("cast").SetCreateFn<CastNvBFloat162FloatKernel>().SetIsMatchedHob(
    (user_op::HobDeviceTag() == "gpu")
    & (user_op::HobDataType("in", 0) == DataType::kBFloat16
       & user_op::HobDataType("out", 0) == DataType::kFloat));

REGISTER_USER_KERNEL("cast_like")
    .SetCreateFn<CastNvBFloat162FloatKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("in", 0) == DataType::kBFloat16
                        & user_op::HobDataType("out", 0) == DataType::kFloat));

REGISTER_USER_KERNEL("cast").SetCreateFn<CastFloat2NvBFloat16Kernel>().SetIsMatchedHob(
    (user_op::HobDeviceTag() == "gpu")
    & (user_op::HobDataType("in", 0) == DataType::kFloat
       & user_op::HobDataType("out", 0) == DataType::kBFloat16));

REGISTER_USER_KERNEL("cast_like")
    .SetCreateFn<CastFloat2NvBFloat16Kernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("in", 0) == DataType::kFloat
                        & user_op::HobDataType("out", 0) == DataType::kBFloat16));

}  // namespace user_op
}  // namespace oneflow

#endif  // defined(CUDA_VERSION) && CUDA_VERSION >= 11000
