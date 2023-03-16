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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

class MluToContiguousKernel final : public user_op::OpKernel {
 public:
  MluToContiguousKernel() = default;
  ~MluToContiguousKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    CnnlTensorDescriptor input_desc, output_desc;
    input_desc.set(in);
    output_desc.set(out);
    OF_CNNL_CHECK(cnnlCopy(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(),
                           in->dptr(), output_desc.desc(), out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("to_contiguous")
    .SetCreateFn<MluToContiguousKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU));

}  // namespace oneflow
