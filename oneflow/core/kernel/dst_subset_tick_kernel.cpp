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

class DstSubsetTickKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DstSubsetTickKernel);
  DstSubsetTickKernel() = default;
  ~DstSubsetTickKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().dst_subset_tick_conf();
  }
};

REGISTER_KERNEL(OperatorConf::kDstSubsetTickConf, DstSubsetTickKernel);

}  // namespace oneflow
