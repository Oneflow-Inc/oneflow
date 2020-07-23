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
#ifndef ONEFLOW_CORE_KERNEL_PRELU_DATA_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PRELU_DATA_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PReluDataGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluDataGradKernel);
  PReluDataGradKernel() = default;
  ~PReluDataGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct PReluDataGradKernelUtil {
  static void Compute(const KernelCtx& ctx, const PReluDataGradOpConf& conf, const Blob* x_blob,
                      const Blob* alpha_blob, const Blob* dy_blob, Blob* dx_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PRELU_DATA_GRAD_KERNEL_H_
