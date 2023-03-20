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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<int Nd, typename T>
class MluMaxPoolKernel final : public user_op::OpKernel {
 public:
  MluMaxPoolKernel() = default;
  ~MluMaxPoolKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
    const std::vector<int32_t>& dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    CHECK_OR_THROW(padding.size() == 2) << "padding size should be 2.";
    CHECK_OR_THROW(kernel_size.size() == 2) << "kernel_size size should be 2.";
    CHECK_OR_THROW(stride.size() == 2) << "stride size should be 2.";
    CHECK_OR_THROW(dilation.size() == 2) << "dilation size should be 2.";
    CHECK_OR_THROW(dilation[0] == 1 && dilation[1] == 1)
        << "cambricon cnnl max pool only supports dilation 1.";

    cnnlTensorLayout_t layout =
        (data_format == "channels_last") ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW;
    cnnlPoolingMode_t mode = CNNL_POOLING_MAX;
    CnnlPoolingDescriptor pooling_desc;
    CnnlTensorDescriptor x_desc, y_desc, indice_desc;
    x_desc.set(x, layout);
    y_desc.set(y, layout);
    indice_desc.set(indice, layout);

    // cnnlPoolingForwardWithIndex requires index_desc->dtype == CNNL_DTYPE_INT32 or
    // CNNL_DTYPE_INT16 But in oneflow/user/ops/max_pool_op.cpp its dtype is set as kInt64.
    // cnnlPoolingForwardWithIndex requires index dtype is int32 for float input,
    // and index dtype is int16 for half input
    auto local_index_dtype = CNNL_DTYPE_INVALID;
    CnnlWorkspace local_index(ctx->stream()->As<ep::MluStream>());
    if (GetDataType<T>::value == DataType::kFloat) {
      local_index_dtype = ConvertToCnnlDataType(kInt32);
      local_index.resize(sizeof(int32_t) * indice->shape_view().elem_cnt());
    } else if (GetDataType<T>::value == DataType::kFloat16) {
      local_index_dtype = ConvertToCnnlDataType(kInt16);
      local_index.resize(sizeof(int16_t) * indice->shape_view().elem_cnt() * 3);
    }
    CnnlTensorDescriptor local_index_desc;
    local_index_desc.set(indice->shape_view().NumAxes(), indice->shape_view().data(),
                         local_index_dtype, layout);

    // calculate paddings
    int pu = padding[0], pd = padding[0], pl = padding[1], pr = padding[1];
    if (ceil_mode) {
      int h_axis = (data_format == "channels_last") ? 1 : 2;
      int w_axis = h_axis + 1;
      int64_t input_h = x->shape_view()[h_axis];
      int64_t input_w = x->shape_view()[w_axis];
      int64_t output_h = y->shape_view()[h_axis];
      int64_t output_w = y->shape_view()[w_axis];
      int diff_height = (output_h - 1) * stride[0] + kernel_size[0] - input_h;
      int diff_width = (output_w - 1) * stride[1] + kernel_size[1] - input_w;
      // If ceil_mode is set to true, the pad needs to be filled up.
      pd = diff_height > padding[0] ? diff_height - padding[0] : 0;
      pr = diff_width > padding[1] ? diff_width - padding[1] : 0;
    }
    pooling_desc.set(mode, kernel_size[0], kernel_size[1], stride[0], stride[1], pu, pd, pl, pr,
                     ceil_mode);

    size_t pooling_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetPoolingWithIndexWorkspaceSize(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* x_desc         */ x_desc.desc(),
        /* y_desc         */ y_desc.desc(),
        /* workspace_size */ &pooling_workspace_size));
    CnnlWorkspace pooling_workspace(ctx->stream()->As<ep::MluStream>(), pooling_workspace_size);

    OF_CNNL_CHECK(cnnlPoolingForwardWithIndex(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* pooling_desc   */ pooling_desc.desc(),
        /* alpha          */ nullptr,
        /* x_desc         */ x_desc.desc(),
        /* x              */ x->dptr(),
        /* beta           */ nullptr,
        /* y_desc         */ y_desc.desc(),
        /* y              */ y->mut_dptr(),
        /* index_desc     */ local_index_desc.desc(),
        /* index          */ local_index.dptr(),
        /* workspace      */ pooling_workspace.dptr(),
        /* workspace_size */ pooling_workspace_size));

    // cast int32/int16 index to int64 index
    CnnlTensorDescriptor int32_index_desc;
    char* int32_index_dptr = reinterpret_cast<char*>(local_index.dptr());
    if (local_index_dtype == CNNL_DTYPE_INT16) {
      int32_index_dptr += sizeof(int16_t) * indice->shape_view().elem_cnt();
      int32_index_desc.set(indice->shape_view().NumAxes(), indice->shape_view().data(),
                           CNNL_DTYPE_INT32, layout);
      OF_CNNL_CHECK(cnnlCastDataType(
          ctx->stream()->As<ep::MluStream>()->cnnl_handle(), local_index_desc.desc(),
          local_index.dptr(), CNNL_CAST_INT16_TO_INT32, int32_index_desc.desc(), int32_index_dptr));
    }
    OF_CNNL_CHECK(cnnlCastDataType(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        (local_index_dtype == CNNL_DTYPE_INT16) ? int32_index_desc.desc() : local_index_desc.desc(),
        int32_index_dptr, CNNL_CAST_INT32_TO_INT64, indice_desc.desc(), indice->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MAX_POOL_MLU_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("max_pool_2d")                                 \
      .SetCreateFn<MluMaxPoolKernel<2, dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MAX_POOL_MLU_KERNEL(float)
REGISTER_MAX_POOL_MLU_KERNEL(float16)

}  // namespace oneflow
