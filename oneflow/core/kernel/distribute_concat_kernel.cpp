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
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace {

void CheckSizeAndCopyBlob(DeviceCtx* ctx, Blob* dst, const Blob* src) {
  dst->CopyValidDataContentFrom(ctx, src);
}

}  // namespace

template<DeviceType device_type>
class DistributeConcatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeConcatKernel);
  DistributeConcatKernel() = default;
  ~DistributeConcatKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const Blob* GetInBlob(std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

template<DeviceType device_type>
void DistributeConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CheckSizeAndCopyBlob(ctx.device_ctx, BnInOp2Blob("out"), GetInBlob(BnInOp2Blob));
}

template<DeviceType device_type>
const Blob* DistributeConcatKernel<device_type>::GetInBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* cur_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != in_blob) {
      CHECK_ISNULL(in_blob);
      in_blob = cur_blob;
    }
  }
  return in_blob;
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDistributeConcatConf, DistributeConcatKernel);

}  // namespace oneflow
