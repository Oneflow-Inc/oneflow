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

namespace oneflow {

namespace {

__global__ void DynamicLossScaleScheduleGpu(const int64_t increment_period, const float multiplier,
                                            const int64_t* count_not_finite, float* loss_scale,
                                            int64_t* good_step_counter) {
  if (*count_not_finite == 0) {
    int64_t cur_good_step_counter = *good_step_counter + 1;
    if (cur_good_step_counter >= increment_period) {
      *loss_scale = static_cast<float>(
          min(static_cast<double>(*loss_scale) * multiplier, static_cast<double>(FLT_MAX)));
      cur_good_step_counter = 0;
    }
    *good_step_counter = cur_good_step_counter;
  } else {
    *good_step_counter = 0;
    *loss_scale = static_cast<float>(max(static_cast<double>(*loss_scale) / multiplier, 1.0));
  }
}

}  // namespace

class DynamicLossScaleScheduleGpuKernel final : public user_op::OpKernel {
 public:
  DynamicLossScaleScheduleGpuKernel() = default;
  ~DynamicLossScaleScheduleGpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* count_not_finite = ctx->Tensor4ArgNameAndIndex("count_not_finite", 0);
    user_op::Tensor* loss_scale = ctx->Tensor4ArgNameAndIndex("loss_scale", 0);
    user_op::Tensor* good_step_counter = ctx->Tensor4ArgNameAndIndex("good_step_counter", 0);
    const auto increment_period = ctx->Attr<int64_t>("increment_period");
    const auto multiplier = ctx->Attr<float>("multiplier");
    DynamicLossScaleScheduleGpu<<<1, 1, 0, ctx->device_ctx()->cuda_stream()>>>(
        increment_period, multiplier, count_not_finite->dptr<int64_t>(),
        loss_scale->mut_dptr<float>(), good_step_counter->mut_dptr<int64_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("dynamic_loss_scale_schedule")
    .SetCreateFn<DynamicLossScaleScheduleGpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu"));

}  // namespace oneflow
