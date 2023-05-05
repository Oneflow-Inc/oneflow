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
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"

namespace oneflow {

namespace {

void PrepareIndexDescAndWorkspace(user_op::KernelComputeContext* ctx,
                                  const std::string& data_format, cnnlDataType_t local_index_dtype,
                                  CnnlTensorDescriptor& index_desc,
                                  CnnlTensorDescriptor& local_index_desc,
                                  CnnlWorkspace& local_index) {
  // prepare index desc and workspace
  const user_op::Tensor* index = ctx->Tensor4ArgNameAndIndex("index", 0);
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  // cnnlPoolingForwardWithIndex requires index_desc->dtype == CNNL_DTYPE_INT32 or
  // CNNL_DTYPE_INT16 But in oneflow/user/ops/max_pool_op.cpp its dtype is set as kInt64.
  // cnnlPoolingForwardWithIndex requires index dtype is int32 for float input,
  // and index dtype is int16 for half input
  if (local_index_dtype == CNNL_DTYPE_INT32) {
    local_index.resize(sizeof(int32_t) * index->shape_view().elem_cnt());
  } else if (local_index_dtype == CNNL_DTYPE_INT16) {
    // elem_cnt * 1 for int16, additional elem_cnt * 2 for casting int16 to int32
    local_index.resize(sizeof(int16_t) * index->shape_view().elem_cnt() * 3);
  } else {
    UNIMPLEMENTED() << "invalid CNNL DType " << local_index_dtype;
  }
  if (data_format == "channels_last") {
    index_desc.set(index->shape_view().size(), index->shape_view().data(),
                   ConvertToCnnlDataType(index->data_type()), layout);
    local_index_desc.set(index->shape_view().NumAxes(), index->shape_view().data(),
                         local_index_dtype, layout);
  } else {
    auto shape = mlu::ComputeShapeNchwToNhwc(Shape(index->shape_view()));
    index_desc.set(index->shape_view().size(), shape.data(),
                   ConvertToCnnlDataType(index->data_type()), CNNL_LAYOUT_NHWC);
    local_index_desc.set(index->shape_view().NumAxes(), shape.data(), local_index_dtype, layout);
  }
}

void ConvertShortIndexToLong(ep::Stream* stream, cnnlDataType_t local_index_dtype,
                             const ShapeView& index_shape, const CnnlTensorDescriptor& index_desc,
                             void* index_ptr, const CnnlTensorDescriptor& local_index_desc,
                             CnnlWorkspace& local_index) {
  // cast int32/int16 index to int64 index
  CnnlTensorDescriptor int32_index_desc;
  char* int32_index_dptr = reinterpret_cast<char*>(local_index.dptr());
  if (local_index_dtype == CNNL_DTYPE_INT16) {
    int32_index_dptr += sizeof(int16_t) * index_shape.elem_cnt();
    int32_index_desc.set(index_shape.NumAxes(), index_shape.data(), CNNL_DTYPE_INT32,
                         CNNL_LAYOUT_NHWC);
    OF_CNNL_CHECK(cnnlCastDataType(
        stream->As<ep::MluStream>()->cnnl_handle(), local_index_desc.desc(), local_index.dptr(),
        CNNL_CAST_INT16_TO_INT32, int32_index_desc.desc(), int32_index_dptr));
  }
  OF_CNNL_CHECK(cnnlCastDataType(
      stream->As<ep::MluStream>()->cnnl_handle(),
      (local_index_dtype == CNNL_DTYPE_INT16) ? int32_index_desc.desc() : local_index_desc.desc(),
      int32_index_dptr, CNNL_CAST_INT32_TO_INT64, index_desc.desc(), index_ptr));
}

void ConvertLongIndexToShort(ep::Stream* stream, cnnlDataType_t local_index_dtype,
                             const ShapeView& index_shape, const CnnlTensorDescriptor& index_desc,
                             const void* index_ptr, const CnnlTensorDescriptor& local_index_desc,
                             CnnlWorkspace& local_index) {
  // cast int64 index to int32/int16 index
  CnnlTensorDescriptor int32_index_desc;
  char* int32_index_dptr = reinterpret_cast<char*>(local_index.dptr());
  if (local_index_dtype == CNNL_DTYPE_INT16) {
    int32_index_dptr += sizeof(int16_t) * index_shape.elem_cnt();
  }
  int32_index_desc.set(index_shape.NumAxes(), index_shape.data(), CNNL_DTYPE_INT32,
                       CNNL_LAYOUT_NHWC);
  OF_CNNL_CHECK(cnnlCastDataType(stream->As<ep::MluStream>()->cnnl_handle(), index_desc.desc(),
                                 index_ptr, CNNL_CAST_INT64_TO_INT32, int32_index_desc.desc(),
                                 int32_index_dptr));
  if (local_index_dtype == CNNL_DTYPE_INT16) {
    OF_CNNL_CHECK(cnnlCastDataType(
        stream->As<ep::MluStream>()->cnnl_handle(), int32_index_desc.desc(), int32_index_dptr,
        CNNL_CAST_INT32_TO_INT16, local_index_desc.desc(), local_index.dptr()));
  }
}

}  // namespace

template<typename T>
class AdaptiveMaxPool2DKernel final : public user_op::OpKernel {
 public:
  AdaptiveMaxPool2DKernel() = default;
  ~AdaptiveMaxPool2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    CnnlTensorDescriptor in_desc, out_desc, index_desc, local_index_desc;
    CnnlWorkspace local_index(ctx->stream()->As<ep::MluStream>());

    cnnlDataType_t dtype = ConvertToCnnlDataType(in_tensor->data_type());
    cnnlDataType_t local_index_dtype =
        in_tensor->data_type() == DataType::kFloat16 ? CNNL_DTYPE_INT16 : CNNL_DTYPE_INT32;
    PrepareIndexDescAndWorkspace(ctx, data_format, local_index_dtype, index_desc, local_index_desc,
                                 local_index);
    if (data_format == "channels_last") {
      in_desc.set(in_tensor->shape_view().NumAxes(), in_tensor->shape_view().data(), dtype,
                  CNNL_LAYOUT_NHWC);
      out_desc.set(out_tensor->shape_view().NumAxes(), out_tensor->shape_view().data(), dtype,
                   CNNL_LAYOUT_NHWC);
      ComputeNHWC(ctx, local_index_desc, local_index, in_desc, in_tensor->dptr(), out_desc,
                  out_tensor->mut_dptr());
      ConvertShortIndexToLong(ctx->stream(), local_index_dtype, index_tensor->shape_view(),
                              index_desc, index_tensor->mut_dptr(), local_index_desc, local_index);
      return;
    }

    size_t tmp_in_workspace_size =
        in_tensor->shape_view().elem_cnt() * GetSizeOfDataType(in_tensor->data_type());
    size_t tmp_out_workspace_size =
        out_tensor->shape_view().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());
    CnnlWorkspace tmp_in_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_in_workspace_size);
    CnnlWorkspace tmp_out_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                         tmp_out_workspace_size);
    mlu::ConvertMemoryFormat(ctx->stream(), in_tensor->shape_view(), in_tensor->data_type(),
                             in_tensor->dptr(), tmp_in_cnnl_workspace.dptr(), MemoryFormat::kNCHW,
                             MemoryFormat::kNHWC);
    void* temp_in_ptr = tmp_in_cnnl_workspace.dptr();
    void* temp_out_ptr = tmp_out_cnnl_workspace.dptr();
    auto in_shape = Shape(in_tensor->shape_view());
    auto out_shape = Shape(out_tensor->shape_view());
    in_shape = mlu::ComputeShapeNchwToNhwc(in_shape);
    out_shape = mlu::ComputeShapeNchwToNhwc(out_shape);
    in_desc.set(in_tensor->shape_view().NumAxes(), in_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    out_desc.set(out_tensor->shape_view().NumAxes(), out_shape.data(), dtype, CNNL_LAYOUT_NHWC);

    ComputeNHWC(ctx, local_index_desc, local_index, in_desc, temp_in_ptr, out_desc, temp_out_ptr);
    mlu::ConvertMemoryFormat(ctx->stream(), out_shape, out_tensor->data_type(), temp_out_ptr,
                             out_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
    auto index_shape = mlu::ComputeShapeNchwToNhwc(Shape(index_tensor->shape_view()));
    ConvertShortIndexToLong(ctx->stream(), local_index_dtype, index_shape, index_desc,
                            index_tensor->mut_dptr(), local_index_desc, local_index);
    // TODO(): convert nhwc index to nchw
  }

  void ComputeNHWC(user_op::KernelComputeContext* ctx, const CnnlTensorDescriptor& local_index_desc,
                   CnnlWorkspace& local_index, const CnnlTensorDescriptor& in_desc,
                   const void* in_ptr, const CnnlTensorDescriptor& out_desc, void* out_ptr) const {
    size_t adaptive_pool2d_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ in_desc.desc(),
        /* mode           */ CNNL_POOLING_MAX,
        /* output_desc    */ out_desc.desc(),
        /* workspace_size */ &adaptive_pool2d_workspace_size));
    CnnlWorkspace adaptive2d_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                            adaptive_pool2d_workspace_size);
    OF_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ in_desc.desc(),
        /* input          */ in_ptr,
        /* mode           */ CNNL_POOLING_MAX,
        /* workspace      */ adaptive2d_cnnl_workspace.dptr(),
        /* workspace_size */ adaptive_pool2d_workspace_size,
        /* output_desc    */ out_desc.desc(),
        /* output         */ out_ptr,
        /* index_desc     */ local_index_desc.desc(),
        /* index          */ local_index.dptr()));
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_POOL2D_MLU_KERNEL(name, dtype)                                    \
  REGISTER_USER_KERNEL(name).SetCreateFn<AdaptiveMaxPool2DKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                        \
      && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_ADAPTIVE_POOL2D_MLU_KERNEL("adaptive_max_pool2d", float)
REGISTER_ADAPTIVE_POOL2D_MLU_KERNEL("adaptive_max_pool2d", float16)

template<typename T>
class AdaptiveMaxPool2DGradKernel final : public user_op::OpKernel {
 public:
  AdaptiveMaxPool2DGradKernel() = default;
  ~AdaptiveMaxPool2DGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    CHECK_EQ(dx_tensor->shape_view().NumAxes(), 4);
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    CnnlTensorDescriptor dy_desc, dx_desc, index_desc, local_index_desc;
    CnnlWorkspace local_index(ctx->stream()->As<ep::MluStream>());

    cnnlDataType_t dtype = ConvertToCnnlDataType(dx_tensor->data_type());
    cnnlDataType_t local_index_dtype =
        dx_tensor->data_type() == DataType::kFloat16 ? CNNL_DTYPE_INT16 : CNNL_DTYPE_INT32;
    PrepareIndexDescAndWorkspace(ctx, data_format, local_index_dtype, index_desc, local_index_desc,
                                 local_index);

    if (data_format == "channels_last") {
      ConvertLongIndexToShort(ctx->stream(), local_index_dtype, index_tensor->shape_view(),
                              index_desc, index_tensor->dptr(), local_index_desc, local_index);

      dy_desc.set(dy_tensor->shape_view().NumAxes(), dy_tensor->shape_view().data(), dtype,
                  CNNL_LAYOUT_NHWC);
      dx_desc.set(dx_tensor->shape_view().NumAxes(), dx_tensor->shape_view().data(), dtype,
                  CNNL_LAYOUT_NHWC);
      ComputeNHWC(ctx, local_index_desc, local_index, dy_desc, dy_tensor->dptr(), dx_desc,
                  dx_tensor->mut_dptr());
      return;
    }

    auto index_shape = mlu::ComputeShapeNchwToNhwc(Shape(index_tensor->shape_view()));
    ConvertLongIndexToShort(ctx->stream(), local_index_dtype, index_shape, index_desc,
                            index_tensor->dptr(), local_index_desc, local_index);

    size_t tmp_dy_workspace_size =
        dy_tensor->shape_view().elem_cnt() * GetSizeOfDataType(dy_tensor->data_type());
    size_t tmp_dx_workspace_size =
        dx_tensor->shape_view().elem_cnt() * GetSizeOfDataType(dx_tensor->data_type());
    CnnlWorkspace tmp_dy_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dy_workspace_size);
    CnnlWorkspace tmp_dx_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dx_workspace_size);
    mlu::ConvertMemoryFormat(ctx->stream(), dy_tensor->shape_view(), dy_tensor->data_type(),
                             dy_tensor->dptr(), tmp_dy_cnnl_workspace.dptr(), MemoryFormat::kNCHW,
                             MemoryFormat::kNHWC);
    void* temp_dy_ptr = tmp_dy_cnnl_workspace.dptr();
    void* temp_dx_ptr = tmp_dx_cnnl_workspace.dptr();
    auto dy_shape = Shape(dy_tensor->shape_view());
    auto dx_shape = Shape(dx_tensor->shape_view());
    dy_shape = mlu::ComputeShapeNchwToNhwc(dy_shape);
    dx_shape = mlu::ComputeShapeNchwToNhwc(dx_shape);
    dy_desc.set(dy_tensor->shape_view().NumAxes(), dy_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    dx_desc.set(dx_tensor->shape_view().NumAxes(), dx_shape.data(), dtype, CNNL_LAYOUT_NHWC);

    ComputeNHWC(ctx, local_index_desc, local_index, dy_desc, temp_dy_ptr, dx_desc, temp_dx_ptr);
    mlu::ConvertMemoryFormat(ctx->stream(), dx_shape, dx_tensor->data_type(), temp_dx_ptr,
                             dx_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
  }

  void ComputeNHWC(user_op::KernelComputeContext* ctx, const CnnlTensorDescriptor& local_index_desc,
                   CnnlWorkspace& local_index, const CnnlTensorDescriptor& dy_desc,
                   const void* dy_ptr, const CnnlTensorDescriptor& dx_desc, void* dx_ptr) const {
    OF_CNNL_CHECK(cnnlAdaptivePoolingBackward(
        /*handle*/ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* y_desc     */ dy_desc.desc(),
        /* y          */ dy_ptr,
        /* index_desc */ local_index_desc.desc(),
        /* index      */ local_index.dptr(),
        /* mode       */ CNNL_POOLING_MAX,
        /* dx_desc    */ dx_desc.desc(),
        /* dx         */ dx_ptr));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_POOL2D_GRAD_MLU_KERNEL(name, dtype)                                   \
  REGISTER_USER_KERNEL(name).SetCreateFn<AdaptiveMaxPool2DGradKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                            \
      && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));
REGISTER_ADAPTIVE_POOL2D_GRAD_MLU_KERNEL("adaptive_max_pool2d_grad", float)
REGISTER_ADAPTIVE_POOL2D_GRAD_MLU_KERNEL("adaptive_max_pool2d_grad", float16)

}  // namespace oneflow
