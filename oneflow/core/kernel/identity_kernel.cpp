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
#include "oneflow/core/ep/include/primitive/memcpy.h"

namespace oneflow {

class IdentityKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentityKernel);
  IdentityKernel() = default;
  ~IdentityKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
  void ForwardHeader(KernelContext* ctx) const override;
};

void IdentityKernel::ForwardDataContent(KernelContext* ctx) const {
  const Blob* in_blob = ctx->BnInOp2Blob("in");
  Blob* out_blob = ctx->BnInOp2Blob("out");
  AutoMemcpy(ctx->stream(), out_blob, in_blob);
}

void IdentityKernel::ForwardHeader(KernelContext* ctx) const {
  ctx->BnInOp2Blob("out")->CopyHeaderFrom(ctx->BnInOp2Blob("in"));
}

REGISTER_KERNEL(OperatorConf::kIdentityConf, IdentityKernel);
REGISTER_KERNEL(OperatorConf::kCopyConf, IdentityKernel);
REGISTER_KERNEL(OperatorConf::kCastToLocalConf, IdentityKernel);
REGISTER_KERNEL(OperatorConf::kCastFromLocalConf, IdentityKernel);
REGISTER_KERNEL(OperatorConf::kBoxingIdentityConf, IdentityKernel);

}  // namespace oneflow
