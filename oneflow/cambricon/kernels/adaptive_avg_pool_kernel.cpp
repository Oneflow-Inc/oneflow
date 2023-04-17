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

template<typename T>
class AdaptiveAvgPool2DKernel final : public user_op::OpKernel {
 public:
  AdaptiveAvgPool2DKernel() = default;
  ~AdaptiveAvgPool2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    cnnlDataType_t dtype = ConvertToCnnlDataType(in_tensor->data_type());
    CnnlTensorDescriptor in_desc, out_desc;

    if (data_format == "channels_last") {
      in_desc.set(in_tensor->shape_view().NumAxes(), in_tensor->shape_view().data(), dtype,
                  CNNL_LAYOUT_NHWC);
      out_desc.set(out_tensor->shape_view().NumAxes(), out_tensor->shape_view().data(), dtype,
                   CNNL_LAYOUT_NHWC);
      ComputeNHWC(ctx, in_desc, in_tensor->dptr(), out_desc, out_tensor->mut_dptr());
      return;
    }

    size_t tmp_in_workspace_size =
        in_tensor->shape_view().elem_cnt() * sizeof(in_tensor->data_type());
    size_t tmp_out_workspace_size =
        out_tensor->shape_view().elem_cnt() * sizeof(out_tensor->data_type());
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

    ComputeNHWC(ctx, in_desc, temp_in_ptr, out_desc, temp_out_ptr);
    mlu::ConvertMemoryFormat(ctx->stream(), out_shape, out_tensor->data_type(), temp_out_ptr,
                             out_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
  }

  void ComputeNHWC(user_op::KernelComputeContext* ctx, const CnnlTensorDescriptor& in_desc,
                   const void* in_ptr, const CnnlTensorDescriptor& out_desc, void* out_ptr) const {
    size_t adaptive_avg_pool2d_workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ in_desc.desc(),
        /* mode           */ CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        /* output_desc    */ out_desc.desc(),
        /* workspace_size */ &adaptive_avg_pool2d_workspace_size));
    CnnlWorkspace adaptive2d_cnnl_workspace(ctx->stream()->As<ep::MluStream>(),
                                            adaptive_avg_pool2d_workspace_size);
    void* adaptive_avg_pool2d_workspace = adaptive2d_cnnl_workspace.dptr();
    OF_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(
        /* handle         */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* input_desc     */ in_desc.desc(),
        /* input          */ in_ptr,
        /* mode           */ CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        /* workspace      */ adaptive_avg_pool2d_workspace,
        /* workspace_size */ adaptive_avg_pool2d_workspace_size,
        /* output_desc    */ out_desc.desc(),
        /* output         */ out_ptr,
        /* index_desc     */ NULL,
        /* index          */ NULL));
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")                         \
      .SetCreateFn<AdaptiveAvgPool2DKernel<dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float)
REGISTER_ADAPTIVE_AVGPOOL2D_MLU_KERNEL(float16)

template<typename T>
class AdaptiveAvgPool2DGradKernel final : public user_op::OpKernel {
 public:
  AdaptiveAvgPool2DGradKernel() = default;
  ~AdaptiveAvgPool2DGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    CHECK_EQ(x_tensor->shape_view().NumAxes(), 4);

    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_last") {
      ComputeNHWC(ctx, dy_tensor, dx_tensor);
      return;
    }

    CnnlTensorDescriptor dy_desc, dx_desc;
    auto dtype = ConvertToCnnlDataType(dy_tensor->data_type());

    size_t tmp_dy_workspace_size =
        dy_tensor->shape_view().elem_cnt() * sizeof(dy_tensor->data_type());
    CnnlWorkspace tmp_dy_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dy_workspace_size);
    void* tmp_dy_ptr = tmp_dy_cnnl_workspace.dptr();

    mlu::ConvertMemoryFormat(ctx->stream(), dy_tensor->shape_view(), dy_tensor->data_type(),
                             dy_tensor->dptr(), tmp_dy_cnnl_workspace.dptr(), MemoryFormat::kNCHW,
                             MemoryFormat::kNHWC);

    auto dy_shape = Shape(dy_tensor->shape_view());
    auto dx_shape = Shape(dx_tensor->shape_view());
    dy_shape = mlu::ComputeShapeNchwToNhwc(dy_shape);
    dx_shape = mlu::ComputeShapeNchwToNhwc(dx_shape);
    dy_desc.set(dy_tensor->shape_view().NumAxes(), dy_shape.data(), dtype, CNNL_LAYOUT_NHWC);
    dx_desc.set(dx_tensor->shape_view().NumAxes(), dx_shape.data(), dtype, CNNL_LAYOUT_NHWC);

    size_t tmp_dx_workspace_size =
        dx_tensor->shape_view().elem_cnt() * sizeof(dy_tensor->data_type());
    CnnlWorkspace tmp_dx_cnnl_workspace(ctx->stream()->As<ep::MluStream>(), tmp_dx_workspace_size);
    void* tmp_dx_ptr = tmp_dx_cnnl_workspace.dptr();

    OF_CNNL_CHECK(cnnlAdaptivePoolingBackward(
        /* handle     */ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* y_desc     */ dy_desc.desc(),
        /* y          */ tmp_dy_ptr,
        /* index_desc */ nullptr,
        /* index      */ nullptr,
        /* mode       */ CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        /* dx_desc    */ dx_desc.desc(),
        /* dx         */ tmp_dx_ptr));

    mlu::ConvertMemoryFormat(ctx->stream(), dx_shape, dx_tensor->data_type(), tmp_dx_ptr,
                             dx_tensor->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
  }

  void ComputeNHWC(user_op::KernelComputeContext* ctx, const user_op::Tensor* dy_tensor,
                   user_op::Tensor* dx_tensor) const {
    auto dtype = ConvertToCnnlDataType(dy_tensor->data_type());
    CnnlTensorDescriptor dy_desc, dx_desc;
    dy_desc.set(dy_tensor->shape_view().NumAxes(), dy_tensor->shape_view().data(), dtype,
                CNNL_LAYOUT_NHWC);
    dx_desc.set(dx_tensor->shape_view().NumAxes(), dx_tensor->shape_view().data(), dtype,
                CNNL_LAYOUT_NHWC);
    OF_CNNL_CHECK(cnnlAdaptivePoolingBackward(
        /*handle*/ ctx->stream()->As<ep::MluStream>()->cnnl_handle(),
        /* y_desc     */ dy_desc.desc(),
        /* y          */ dy_tensor->dptr(),
        /* index_desc */ nullptr,
        /* index      */ nullptr,
        /* mode       */ CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        /* dx_desc    */ dx_desc.desc(),
        /* dx         */ dx_tensor->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_AVGPOOL2D_GRAD_MLU_KERNEL(dtype)            \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")                    \
      .SetCreateFn<AdaptiveAvgPool2DGradKernel<dtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));
REGISTER_ADAPTIVE_AVGPOOL2D_GRAD_MLU_KERNEL(float)
REGISTER_ADAPTIVE_AVGPOOL2D_GRAD_MLU_KERNEL(float16)

}  // namespace oneflow
