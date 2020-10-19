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
#ifndef ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropMdUpdateKernel);
  RMSPropMdUpdateKernel() = default;
  ~RMSPropMdUpdateKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void UpdateModel(DeviceCtx* ctx, T weight_decay, const int64_t* train_step,
                   const float* learning_rate,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernelUtil final {
 public:
  // mean_square = (1 - decay_rate) * model_diff ^ 2 + decay_rate * mean_square
  // if (centered) {
  //    mean_gradient = (1 - decay_rate) * model_diff + decay_rate * mean_gradient
  //    denom_t = mean_square - mean_gradient ^ 2
  // } else {
  //    denom_t = mean_square
  // }
  // model = model - learning_rate * model_diff / sqrt(denom_t + epsilon)
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const int64_t* train_step,
                          const float* learning_rate, T decay_rate, T epsilon, bool centered,
                          T weight_decay, const T* model_diff, T* model, T* mean_square,
                          T* mean_gradient);
};

DECLARE_MDUPDT_KERNEL_CREATOR(RMSProp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
