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

#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class MluTopkKernel final : public user_op::OpKernel {
 public:
  MluTopkKernel() = default;
  ~MluTopkKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t k = ctx->Attr<int32_t>("k");
    const bool sorted = ctx->Attr<bool>("sorted");

    CnnlTensorDescriptor in_desc(in), out_desc(out);

    // out_value saves the max value
    CnnlTensorDescriptor out_value_desc;
    out_value_desc.set(out, ConvertToCnnlDataType(in->data_type()));
    CnnlWorkspace out_value(ctx->stream()->As<ep::MluStream>(),
                            GetSizeOfDataType(in->data_type()) * out->shape_view().elem_cnt());

    // out_indices saves the index of max value in int32
    CnnlTensorDescriptor out_indices_desc;
    out_indices_desc.set(out, CNNL_DTYPE_INT32);
    CnnlWorkspace out_indices(ctx->stream()->As<ep::MluStream>(),
                              GetSizeOfDataType(kInt32) * out->shape_view().elem_cnt());

    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetTopKTensorWorkspaceSize(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ in_desc.desc(),
        /* k              */ static_cast<int>(k),
        /* dim            */ static_cast<int>(in->shape_view().NumAxes() - 1),
        /* largest        */ true,
        /* output_desc    */ out_value_desc.desc(),
        /* index_desc     */ out_indices_desc.desc(),
        /* workspace_size */ &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);

    OF_CNNL_CHECK(cnnlTopKTensor_v3(
        /* handle            */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc        */ in_desc.desc(),
        /* input             */ in->dptr(),
        /* k                 */ static_cast<int>(k),
        /* dim               */ static_cast<int>(in->shape_view().NumAxes() - 1),
        /* largest           */ true,
        /* sorted            */ sorted,
        /* lower_index_first */ true,
        /* workspace         */ workspace.dptr(),
        /* workspace_size    */ workspace_size,
        /* output_desc       */ out_value_desc.desc(),
        /* output            */ out_value.dptr(),
        /* index_desc        */ out_indices_desc.desc(),
        /* index             */ out_indices.dptr()));

    OF_CNNL_CHECK(cnnlCastDataType(
        /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc  */ out_indices_desc.desc(),
        /* input       */ out_indices.dptr(),
        /* cast_type   */ CNNL_CAST_INT32_TO_INT64,
        /* output_desc */ out_desc.desc(),
        /* output      */ out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TOPK_MLU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("top_k").SetCreateFn<MluTopkKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                 \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_TOPK_MLU_KERNEL(float)
REGISTER_TOPK_MLU_KERNEL(float16)
REGISTER_TOPK_MLU_KERNEL(int8_t)
REGISTER_TOPK_MLU_KERNEL(uint8_t)
REGISTER_TOPK_MLU_KERNEL(int32_t)

}  // namespace oneflow
