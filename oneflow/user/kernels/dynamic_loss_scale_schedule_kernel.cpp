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

class DynamicLossScaleScheduleCpuKernel final : public user_op::OpKernel {
 public:
  DynamicLossScaleScheduleCpuKernel() = default;
  ~DynamicLossScaleScheduleCpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* count_not_finite =
        ctx->Tensor4ArgNameAndIndex("count_not_finite", 0)->dptr<int64_t>();
    auto* loss_scale = ctx->Tensor4ArgNameAndIndex("loss_scale", 0)->mut_dptr<float>();
    auto* good_step_counter =
        ctx->Tensor4ArgNameAndIndex("good_step_counter", 0)->mut_dptr<int64_t>();
    const auto increment_period = ctx->Attr<int64_t>("increment_period");
    const auto multiplier = ctx->Attr<float>("multiplier");
    if (*count_not_finite == 0) {
      int64_t cur_good_step_counter = *good_step_counter + 1;
      if (cur_good_step_counter >= increment_period) {
        const double old_loss_scale = *loss_scale;
        const double new_loss_scale =
            std::min(old_loss_scale * multiplier, static_cast<double>(FLT_MAX));
        *loss_scale = static_cast<float>(new_loss_scale);
        cur_good_step_counter = 0;
        LOG(INFO) << "In past " << increment_period
                  << " steps, there are no nan or inf in gradients, so we increase loss_scale from "
                  << old_loss_scale << " to " << new_loss_scale;
      }
      *good_step_counter = cur_good_step_counter;
    } else {
      *good_step_counter = 0;
      const double old_loss_scale = *loss_scale;
      const double new_loss_scale = std::max(old_loss_scale / multiplier, 1.0);
      *loss_scale = static_cast<float>(new_loss_scale);
      LOG(INFO) << "There are nan or inf in gradients, so we decrease loss_scale from "
                << old_loss_scale << " to " << new_loss_scale;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("dynamic_loss_scale_schedule")
    .SetCreateFn<DynamicLossScaleScheduleCpuKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu"));

}  // namespace oneflow
