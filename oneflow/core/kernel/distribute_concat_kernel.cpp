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
class DistributeConcatKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeConcatKernel);
  DistributeConcatKernel() = default;
  ~DistributeConcatKernel() = default;

 private:
  void ForwardDataContent(const KernelContext* ctx) const override;
  const Blob* GetInBlob(const KernelContext* ctx) const;
};

template<DeviceType device_type>
void DistributeConcatKernel<device_type>::ForwardDataContent(const KernelContext* ctx) const {
  CheckSizeAndCopyBlob(ctx->device_ctx(), ctx->BnInOp2Blob("out"), GetInBlob(ctx));
}

template<DeviceType device_type>
const Blob* DistributeConcatKernel<device_type>::GetInBlob(const KernelContext* ctx) const {
  const Blob* in_blob = nullptr;
  FOR_RANGE(int, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* cur_blob = ctx->BnInOp2Blob(this->op_attribute().input_bns().Get(i));
    if (cur_blob != nullptr && cur_blob != in_blob) {
      CHECK_ISNULL(in_blob);
      in_blob = cur_blob;
    }
  }
  return in_blob;
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDistributeConcatConf, DistributeConcatKernel);

}  // namespace oneflow
