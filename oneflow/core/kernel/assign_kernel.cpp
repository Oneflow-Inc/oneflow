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
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class AssignKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AssignKernel);
  AssignKernel() = default;
  ~AssignKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
void AssignKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("ref")->CopyValidDataContentFrom(ctx.device_ctx, BnInOp2Blob("value"));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAssignConf, DeviceType::kCPU,
                            AssignKernel<DeviceType::kCPU>);
#ifdef WITH_CUDA
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAssignConf, DeviceType::kGPU,
                            AssignKernel<DeviceType::kGPU>);
#endif

}  // namespace oneflow
