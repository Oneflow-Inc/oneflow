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
#include "oneflow/core/ep/include/primitive/memset.h"

namespace oneflow {

class BoxingZerosKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingZerosKernel);
  BoxingZerosKernel() = default;
  ~BoxingZerosKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  void ForwardDataContent(KernelContext* ctx) const override;

  std::unique_ptr<ep::primitive::Memset> primitive_;
};

void BoxingZerosKernel::VirtualKernelInit(KernelContext* ctx) {
  primitive_ =
      ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->stream()->device_type());
  CHECK(primitive_);
}

void BoxingZerosKernel::ForwardDataContent(KernelContext* ctx) const {
  Blob* out = ctx->BnInOp2Blob("out");
  primitive_->Launch(ctx->stream(), out->mut_dptr(), 0, out->ByteSizeOfBlobBody());
}

REGISTER_KERNEL(OperatorConf::kBoxingZerosConf, BoxingZerosKernel);

}  // namespace oneflow
