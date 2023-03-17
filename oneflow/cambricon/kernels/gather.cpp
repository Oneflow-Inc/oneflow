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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

class MluGatherKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  MluGatherKernel() = default;
  ~MluGatherKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t axis = ctx->Attr<int64_t>("axis");

    if (out->shape_view().elem_cnt() == 0) { return; }

    CnnlTensorDescriptor in_desc(in), indices_desc(indices), out_desc(out);
    OF_CNNL_CHECK(cnnlIndexSelect(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), axis,
                                  in_desc.desc(), in->dptr(), indices_desc.desc(), indices->dptr(),
                                  out_desc.desc(), out->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("gather").SetCreateFn<MluGatherKernel>().SetIsMatchedHob(
    (user_op::HobDeviceType() == DeviceType::kMLU));

}  // namespace oneflow