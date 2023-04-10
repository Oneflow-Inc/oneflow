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
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"
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

    const DataType data_type = x->data_type();
    auto cnnl_data_type = ConvertToCnnlDataType(data_type);
    cnnlTensorLayout_t layout =
        (data_format == "channels_last") ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW;
    cnnlPoolingMode_t mode = CNNL_POOLING_MAX;
    CnnlPoolingDescriptor pooling_desc;
    CnnlTensorDescriptor x_desc, y_desc, indice_desc;
    x_desc.set(x->shape_view().size(), x->shape_view().data(), cnnl_data_type, layout);
    y_desc.set(y->shape_view().size(), y->shape_view().data(), cnnl_data_type, layout);
    indice_desc.set(indice->shape_view().size(), indice->shape_view().data(),
                    ConvertToCnnlDataType(indice->data_type()), layout);

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

template<int Nd, typename T>
class MluMaxPoolGradKernel final : public user_op::OpKernel {
 public:
  MluMaxPoolGradKernel() = default;
  ~MluMaxPoolGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    auto data_type = x->data_type();
    auto cnnl_data_type = ConvertToCnnlDataType(data_type);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    cnnlPoolingMode_t mode = CNNL_POOLING_MAX;

    CnnlTensorDescriptor x_desc, indice_desc, dy_desc, dx_desc;
    auto shape = Shape(x->shape_view());
    auto indice_shape = Shape(indice->shape_view());
    auto dy_shape = Shape(dy->shape_view());
    const void* temp_x = x->dptr();
    const void* temp_indice = indice->dptr();
    const void* temp_dy = dy->dptr();
    void* temp_dx = dx->mut_dptr();
    CnnlWorkspace temp_x_workspace(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_indice_workspace(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_dy_workspace(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_dx_workspace(ctx->stream()->As<ep::MluStream>());

    if (data_format != "channels_last") {
      size_t element_size = GetSizeOfDataType(data_type);
      shape = mlu::ComputeShapeNchwToNhwc(shape);
      indice_shape = mlu::ComputeShapeNchwToNhwc(indice_shape);
      dy_shape = mlu::ComputeShapeNchwToNhwc(dy_shape);
      temp_x_workspace.resize(shape.elem_cnt() * element_size);
      temp_indice_workspace.resize(indice_shape.elem_cnt()
                                   * GetSizeOfDataType(indice->data_type()));
      temp_dy_workspace.resize(dy_shape.elem_cnt() * element_size);
      temp_dx_workspace.resize(shape.elem_cnt() * element_size);
      // convert x to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), x->shape_view(), data_type, x->dptr(),
                               temp_x_workspace.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      // convert dy to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), dy->shape_view(), data_type, dy->dptr(),
                               temp_dy_workspace.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      // convert indice to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), indice->shape_view(), indice->data_type(),
                               indice->dptr(), temp_indice_workspace.dptr(), MemoryFormat::kNCHW,
                               MemoryFormat::kNHWC);
      temp_x = temp_x_workspace.dptr();
      temp_indice = temp_indice_workspace.dptr();
      temp_dy = temp_dy_workspace.dptr();
      temp_dx = temp_dx_workspace.dptr();
    }
    x_desc.set(shape.size(), shape.data(), cnnl_data_type, layout);
    dy_desc.set(dy_shape.size(), dy_shape.data(), cnnl_data_type, layout);
    indice_desc.set(indice_shape.size(), indice_shape.data(),
                    ConvertToCnnlDataType(indice->data_type()), layout);
    dx_desc.set(shape.size(), shape.data(), cnnl_data_type, layout);

    // cnnlPoolingBackward requires y_desc is int32/int16, which is int64 in oneflow op
    auto local_index_dtype = CNNL_DTYPE_INVALID;
    CnnlWorkspace local_index(ctx->stream()->As<ep::MluStream>());
    if (GetDataType<T>::value == DataType::kFloat) {
      local_index_dtype = CNNL_DTYPE_INT32;
      local_index.resize(sizeof(int32_t) * indice_shape.elem_cnt());
    } else if (GetDataType<T>::value == DataType::kFloat16) {
      local_index_dtype = CNNL_DTYPE_INT16;
      local_index.resize(sizeof(int16_t) * indice_shape.elem_cnt() * 3);
    }
    CnnlTensorDescriptor local_index_desc;
    local_index_desc.set(indice_shape.NumAxes(), indice_shape.data(), local_index_dtype, layout);

    if (local_index_dtype == CNNL_DTYPE_INT16) {
      CnnlTensorDescriptor int32_index_desc;
      int32_index_desc.set(indice_shape.NumAxes(), indice_shape.data(), CNNL_DTYPE_INT32, layout);
      char* int32_index_dptr =
          reinterpret_cast<char*>(local_index.dptr()) + sizeof(int16_t) * indice_shape.elem_cnt();
      OF_CNNL_CHECK(cnnlCastDataType(
          /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
          /* input_desc  */ indice_desc.desc(),
          /* input       */ temp_indice,
          /* cast_type   */ CNNL_CAST_INT64_TO_INT32,
          /* output_desc */ int32_index_desc.desc(),
          /* output      */ int32_index_dptr));

      OF_CNNL_CHECK(cnnlCastDataType(
          /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
          /* input_desc  */ int32_index_desc.desc(),
          /* input       */ int32_index_dptr,
          /* cast_type   */ CNNL_CAST_INT32_TO_INT16,
          /* output_desc */ local_index_desc.desc(),
          /* output      */ local_index.dptr()));
    } else {
      OF_CNNL_CHECK(cnnlCastDataType(
          /* handle      */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
          /* input_desc  */ indice_desc.desc(),
          /* input       */ temp_indice,
          /* cast_type   */ CNNL_CAST_INT64_TO_INT32,
          /* output_desc */ local_index_desc.desc(),
          /* output      */ local_index.dptr()));
    }
    CnnlPoolingDescriptor pooling_desc;
    // calculate paddings
    int pu = padding[0], pd = padding[0], pl = padding[1], pr = padding[1];
    pooling_desc.set(mode, kernel_size[0], kernel_size[1], stride[0], stride[1], pu, pd, pl, pr,
                     ceil_mode);

    OF_CNNL_CHECK(cnnlPoolingBackward(
        /* handle       */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* pooling_desc */ pooling_desc.desc(),
        /* alpha        */ nullptr,
        /* y_desc       */ local_index_desc.desc(),
        /* y            */ local_index.dptr(),
        /* diff_y_desc  */ dy_desc.desc(),
        /* diff_y       */ temp_dy,
        /* x_desc       */ x_desc.desc(),
        /* x            */ temp_x,
        /* beta         */ nullptr,
        /* diff_x_desc  */ dx_desc.desc(),
        /* diff_x       */ temp_dx));

    if (data_format != "channels_last") {
      // convert dx to NCHW
      mlu::ConvertMemoryFormat(ctx->stream(), shape, data_type, temp_dx, dx->mut_dptr(),
                               MemoryFormat::kNHWC, MemoryFormat::kNCHW);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MAX_POOL_GRAD_MLU_KERNEL(x_dtype, indice_dtype)                        \
  REGISTER_USER_KERNEL("max_pool_2d_grad")                                              \
      .SetCreateFn<MluMaxPoolGradKernel<2, x_dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                   \
                       && (user_op::HobDataType("x", 0) == GetDataType<x_dtype>::value) \
                       && user_op::HobDataType("indice", 0) == GetDataType<indice_dtype>::value);

REGISTER_MAX_POOL_GRAD_MLU_KERNEL(float, int64_t)
REGISTER_MAX_POOL_GRAD_MLU_KERNEL(float16, int64_t)

}  // namespace oneflow
