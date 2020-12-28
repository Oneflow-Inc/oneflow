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
#ifndef ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

class DecodeOFRecordKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOFRecordKernel);
  DecodeOFRecordKernel() = default;
  ~DecodeOFRecordKernel() = default;

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  void VirtualKernelInit() override;
  int32_t NextRandomInt() const;

  std::unique_ptr<std::mt19937> random_seed_gen_;
  std::unique_ptr<std::uniform_int_distribution<int32_t>> distribution_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
