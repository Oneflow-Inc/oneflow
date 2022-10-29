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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {
template<typename T>
__global__ void AmpUpdateScaleImpl(T* current_scale, int* growth_tracker, const T* found_inf,
                                   double growth_factor, double backoff_factor,
                                   int64_t growth_interval) {
  if (*found_inf) {
    *current_scale = (*current_scale) * backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    if (successful == growth_interval) {
      *current_scale = (*current_scale) * growth_factor;
      *growth_tracker = 0;
    } else {
      *growth_tracker = successful;
    }
  }
}
};  // namespace

template<typename T>
class AMPUpdateScaleGpuKernel final : public user_op::OpKernel {
 public:
  AMPUpdateScaleGpuKernel() = default;
  ~AMPUpdateScaleGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* current_scale = ctx->Tensor4ArgNameAndIndex("current_scale", 0);
    user_op::Tensor* growth_tracker = ctx->Tensor4ArgNameAndIndex("growth_tracker", 0);
    const user_op::Tensor* found_inf = ctx->Tensor4ArgNameAndIndex("found_inf", 0);

    const double growth_factor = ctx->Attr<double>("growth_factor");
    const double backoff_factor = ctx->Attr<double>("backoff_factor");
    const int64_t growth_interval = ctx->Attr<int64_t>("growth_interval");

    AmpUpdateScaleImpl<T><<<1, 1, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        current_scale->mut_dptr<T>(), growth_tracker->mut_dptr<int>(), found_inf->dptr<T>(),
        growth_factor, backoff_factor, growth_interval);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class AMPForEachNonFiniteCheckAndUnscaleGpuKernel final : public user_op::OpKernel {
 public:
  AMPForEachNonFiniteCheckAndUnscaleGpuKernel() = default;
  ~AMPForEachNonFiniteCheckAndUnscaleGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {}
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_AMP_CUDA_KERNELS(dtype)                                                         \
  REGISTER_USER_KERNEL("amp_update_scale")                                                       \
      .SetCreateFn<AMPUpdateScaleGpuKernel<dtype>>()                                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                           \
                       && (user_op::HobDataType("next_scale", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("amp_non_finite_check_and_unscale")                                       \
      .SetCreateFn<AMPForEachNonFiniteCheckAndUnscaleGpuKernel<dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                           \
                       && (user_op::HobDataType("scaled_grads", 0) == GetDataType<dtype>::value));

REGISTER_AMP_CUDA_KERNELS(float)
REGISTER_AMP_CUDA_KERNELS(double)

}  // namespace oneflow
