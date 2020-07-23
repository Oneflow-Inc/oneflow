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
#ifndef ONEFLOW_CORE_KERNEL_SINK_TICK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SINK_TICK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class SinkTickKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SinkTickKernel);
  SinkTickKernel() = default;
  ~SinkTickKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().source_tick_conf();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SINK_TICK_KERNEL_H_
