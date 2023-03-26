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
#include "oneflow/core/framework/framework.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

class MluDimGatherKernel final : public user_op::OpKernel {
 public:
  MluDimGatherKernel() = default;
  ~MluDimGatherKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    const user_op::Tensor* index = ctx->Tensor4ArgNameAndIndex("index", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");

    CnnlTensorDescriptor in_desc(in), index_desc(index), out_desc(out);
    OF_CNNL_CHECK(cnnlGather(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), dim, in_desc.desc(),
                             in->dptr(), index_desc.desc(), index->dptr(), out_desc.desc(),
                             out->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("dim_gather")
    .SetCreateFn<MluDimGatherKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU));

}  // namespace oneflow
