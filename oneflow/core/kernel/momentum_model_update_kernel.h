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
#ifndef ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MomentumMdUpdateKernel);
  MomentumMdUpdateKernel() = default;
  ~MomentumMdUpdateKernel() = default;

 protected:
  const PbMessage& GetCustomizedOpConf() const override;

 private:
  void UpdateModel(DeviceCtx* ctx, T weight_decay, const int64_t* train_step,
                   const float* learning_rate,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  bool IsWeightDecaySupported() override { return true; }
};

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, T beta, const int64_t* train_step,
                          const float* learning_rate, T weight_decay, const T* model_diff, T* model,
                          T* momentum);
};

DECLARE_MDUPDT_KERNEL_CREATOR(Momentum);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
