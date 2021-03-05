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

namespace {

template<DeviceType device_type>
class InputKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InputKernel);
  InputKernel() = default;
  ~InputKernel() = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  void ForwardHeader(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
};

}  // namespace

ADD_DEVICE_TYPE_KERNEL_CREATOR_INCLUDING_FAKE(OperatorConf::kInputConf, InputKernel);

}  // namespace oneflow
