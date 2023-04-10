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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"

namespace oneflow {

class MluConvertMemoryFormatKernel final : public user_op::OpKernel {
 public:
  MluConvertMemoryFormatKernel() = default;
  ~MluConvertMemoryFormatKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& memory_format = ctx->Attr<MemoryFormat>("memory_format");
    CHECK_EQ_OR_THROW(out->memory_format(), memory_format)
        << "output memory format should be " << MemoryFormat_Name(memory_format);

    mlu::ConvertMemoryFormat(ctx->stream(), in, out, in->memory_format(), out->memory_format());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("convert_memory_format")
    .SetCreateFn<MluConvertMemoryFormatKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU));

}  // namespace oneflow
