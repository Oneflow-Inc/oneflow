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
#include "oneflow/core/cuda/elementwise.cuh"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename T, typename U>
struct CastFunctor {
  OF_DEVICE_FUNC T operator()(U x) const { return static_cast<T>(x); }
};

}  // namespace

template<typename T, typename U>
class CastBFloat16Kernel final : public OpKernel {
 public:
  CastBFloat16Kernel() = default;
  ~CastBFloat16Kernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t n = in->shape().elem_cnt();
    OF_CUDA_CHECK((cuda::elementwise::Unary<CastFunctor<T, U>, T, U>(
        CastFunctor<T, U>(), n, reinterpret_cast<T*>(out->mut_dptr()),
        reinterpret_cast<const U*>(in->dptr()), ctx->device_ctx()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("cast").SetCreateFn<CastBFloat16Kernel<float, nv_bfloat16>>().SetIsMatchedHob(
    (user_op::HobDeviceTag() == "gpu")
    & (user_op::HobDataType("in", 0) == DataType::kBFloat16
       & user_op::HobDataType("out", 0) == DataType::kFloat));

REGISTER_USER_KERNEL("cast_like")
    .SetCreateFn<CastBFloat16Kernel<float, nv_bfloat16>>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("in", 0) == DataType::kBFloat16
                        & user_op::HobDataType("out", 0) == DataType::kFloat));

REGISTER_USER_KERNEL("cast").SetCreateFn<CastBFloat16Kernel<nv_bfloat16, float>>().SetIsMatchedHob(
    (user_op::HobDeviceTag() == "gpu")
    & (user_op::HobDataType("in", 0) == DataType::kFloat
       & user_op::HobDataType("out", 0) == DataType::kBFloat16));

REGISTER_USER_KERNEL("cast_like")
    .SetCreateFn<CastBFloat16Kernel<nv_bfloat16, float>>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("in", 0) == DataType::kFloat
                        & user_op::HobDataType("out", 0) == DataType::kBFloat16));

}  // namespace user_op
}  // namespace oneflow

#endif  // defined(CUDA_VERSION) && CUDA_VERSION >= 11000
