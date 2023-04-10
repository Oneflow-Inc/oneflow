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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/kernels/convert_memory_format_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

void UpdateConvParams(std::vector<int32_t>* paddings, std::vector<int32_t>* strides,
                      std::vector<int32_t>* dilation_rates, int kernel_size) {
  size_t paddings_size = paddings->size();
  size_t strides_size = strides->size();
  size_t dilation_rates_size = dilation_rates->size();

  auto repeat_assign = [](std::vector<int32_t>* vec, int32_t value) {
    for (int i = 0; i < vec->size(); ++i) { (*vec)[i] = value; }
  };
  if (paddings_size == 1) {
    paddings->resize(kernel_size);
    repeat_assign(paddings, (*paddings)[0]);
  }
  if (strides_size == 1) {
    strides->resize(kernel_size);
    repeat_assign(strides, (*strides)[0]);
  }
  if (dilation_rates_size == 1) {
    dilation_rates->resize(kernel_size);
    repeat_assign(dilation_rates, (*dilation_rates)[0]);
  }
}

template<typename T>
class Conv2DKernel final : public user_op::OpKernel {
 public:
  Conv2DKernel() = default;
  ~Conv2DKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int& groups = ctx->Attr<int32_t>("groups");
    std::vector<int32_t> paddings = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation_rates = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    auto in_shape = Shape(in->shape_view());
    auto weight_shape = Shape(weight->shape_view());
    auto out_shape = Shape(out->shape_view());
    const void* input_ptr = in->dptr();
    const void* weight_ptr = weight->dptr();
    void* output_ptr = out->mut_dptr();

    int32_t kernel_size = in_shape.NumAxes() - 2;
    UpdateConvParams(&paddings, &strides, &dilation_rates, kernel_size);

    auto data_type = in->data_type();
    auto cnnl_data_type = ConvertToCnnlDataType(data_type);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDescriptor input_desc, weight_desc, bias_desc, output_desc;
    CnnlConvolutionDescriptor conv_desc;
    conv_desc.set(in_shape.NumAxes(), strides.data(), paddings.data(), dilation_rates.data(),
                  groups, cnnl_data_type);

    CnnlWorkspace temp_input(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_weight(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_output(ctx->stream()->As<ep::MluStream>());

    if (data_format != "channels_last") {
      size_t element_size = GetSizeOfDataType(data_type);
      in_shape = mlu::ComputeShapeNchwToNhwc(in_shape);
      weight_shape = mlu::ComputeShapeNchwToNhwc(weight_shape);
      out_shape = mlu::ComputeShapeNchwToNhwc(out_shape);
      temp_input.resize(in_shape.elem_cnt() * element_size);
      temp_weight.resize(weight_shape.elem_cnt() * element_size);
      temp_output.resize(out_shape.elem_cnt() * element_size);
      // convert input to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), in->shape_view(), data_type, in->dptr(),
                               temp_input.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      // convert weight to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), weight->shape_view(), data_type, weight->dptr(),
                               temp_weight.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      input_ptr = temp_input.dptr();
      weight_ptr = temp_weight.dptr();
      output_ptr = temp_output.dptr();
    }
    input_desc.set(in_shape.NumAxes(), in_shape.data(), cnnl_data_type, layout);
    weight_desc.set(weight_shape.NumAxes(), weight_shape.data(), cnnl_data_type, layout);
    output_desc.set(out_shape.NumAxes(), out_shape.data(), cnnl_data_type, layout);

    const void* bias_ptr = nullptr;
    if (bias) {
      int64_t bias_sizes[1] = {bias->shape_view().elem_cnt()};
      bias_desc.set(1, bias_sizes, cnnl_data_type, layout);
      bias_ptr = bias->dptr();
    }
    cnnlConvolutionForwardAlgo_t algo;
    OF_CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), conv_desc.desc(), input_desc.desc(),
        weight_desc.desc(), output_desc.desc(), CNNL_CONVOLUTION_FWD_FASTEST, &algo));

    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), input_desc.desc(), weight_desc.desc(),
        output_desc.desc(), bias_desc.desc(), conv_desc.desc(), algo, &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);

    OF_CNNL_CHECK(cnnlConvolutionForward(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), conv_desc.desc(), algo, nullptr,
        input_desc.desc(), input_ptr, weight_desc.desc(), weight_ptr, bias_desc.desc(), bias_ptr,
        workspace.dptr(), workspace_size, nullptr, output_desc.desc(), output_ptr));

    if (data_format != "channels_last") {
      // convert output to NCHW
      mlu::ConvertMemoryFormat(ctx->stream(), out_shape, data_type, output_ptr, out->mut_dptr(),
                               MemoryFormat::kNHWC, MemoryFormat::kNCHW);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONV2D_MLU_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("conv2d").SetCreateFn<Conv2DKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                 \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CONV2D_MLU_KERNEL(float)
REGISTER_CONV2D_MLU_KERNEL(float16)

template<typename T>
class ConvDataGradKernel final : public user_op::OpKernel {
 public:
  ConvDataGradKernel() = default;
  ~ConvDataGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* filter = ctx->Tensor4ArgNameAndIndex("filter", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    if (dx->shape_view().elem_cnt() == 0) return;

    const int& groups = ctx->Attr<int32_t>("groups");
    std::vector<int32_t> paddings = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation_rates = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    auto dy_shape = Shape(dy->shape_view());
    auto filter_shape = Shape(filter->shape_view());
    auto dx_shape = Shape(dx->shape_view());
    const void* dy_ptr = dy->dptr();
    const void* filter_ptr = filter->dptr();
    void* dx_ptr = dx->mut_dptr();

    int32_t kernel_dims = dx_shape.NumAxes() - 2;
    UpdateConvParams(&paddings, &strides, &dilation_rates, kernel_dims);

    auto data_type = dy->data_type();
    auto cnnl_data_type = ConvertToCnnlDataType(data_type);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDescriptor dy_desc, filter_desc, dx_desc;
    CnnlConvolutionDescriptor conv_desc;
    conv_desc.set(dx_shape.NumAxes(), strides.data(), paddings.data(), dilation_rates.data(),
                  groups, cnnl_data_type);

    CnnlWorkspace temp_dy(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_filter(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_dx(ctx->stream()->As<ep::MluStream>());

    if (data_format != "channels_last") {
      size_t element_size = GetSizeOfDataType(dx->data_type());
      dy_shape = mlu::ComputeShapeNchwToNhwc(dy_shape);
      filter_shape = mlu::ComputeShapeNchwToNhwc(filter_shape);
      dx_shape = mlu::ComputeShapeNchwToNhwc(dx_shape);
      temp_dy.resize(dy_shape.elem_cnt() * element_size);
      temp_filter.resize(filter_shape.elem_cnt() * element_size);
      temp_dx.resize(dx_shape.elem_cnt() * element_size);
      // convert dy to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), dy->shape_view(), data_type, dy->dptr(),
                               temp_dy.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      // convert filter to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), filter->shape_view(), data_type, filter->dptr(),
                               temp_filter.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      dy_ptr = temp_dy.dptr();
      filter_ptr = temp_filter.dptr();
      dx_ptr = temp_dx.dptr();
    }
    dy_desc.set(dy_shape.NumAxes(), dy_shape.data(), cnnl_data_type, layout);
    filter_desc.set(filter_shape.NumAxes(), filter_shape.data(), cnnl_data_type, layout);
    dx_desc.set(dx_shape.NumAxes(), dx_shape.data(), cnnl_data_type, layout);

    cnnlConvolutionBwdDataAlgo_t algo;
    OF_CNNL_CHECK(cnnlGetConvolutionBackwardDataAlgorithm(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), filter_desc.desc(), dy_desc.desc(),
        conv_desc.desc(), dx_desc.desc(), CNNL_CONVOLUTION_BWD_DATA_FASTEST, &algo));

    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetConvolutionBackwardDataWorkspaceSize(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), filter_desc.desc(), dy_desc.desc(),
        conv_desc.desc(), dx_desc.desc(), algo, &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);

    OF_CNNL_CHECK(cnnlConvolutionBackwardData(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, filter_desc.desc(), filter_ptr,
        dy_desc.desc(), dy_ptr, conv_desc.desc(), algo, workspace.dptr(), workspace_size, nullptr,
        dx_desc.desc(), dx_ptr));

    if (data_format != "channels_last") {
      // convert dx to NCHW
      mlu::ConvertMemoryFormat(ctx->stream(), dx_shape, data_type, dx_ptr, dx->mut_dptr(),
                               MemoryFormat::kNHWC, MemoryFormat::kNCHW);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONV_DATA_GRAD_MLU_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("conv_data_grad")                              \
      .SetCreateFn<ConvDataGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_CONV_DATA_GRAD_MLU_KERNEL(float)
REGISTER_CONV_DATA_GRAD_MLU_KERNEL(float16)

template<typename T>
class ConvFilterGradKernel final : public user_op::OpKernel {
 public:
  ConvFilterGradKernel() = default;
  ~ConvFilterGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* filter_diff = ctx->Tensor4ArgNameAndIndex("filter_diff", 0);

    if (x->shape_view().elem_cnt() == 0) {
      Memset<DeviceType::kMLU>(
          ctx->stream(), filter_diff->mut_dptr(), 0,
          filter_diff->shape_view().elem_cnt() * GetSizeOfDataType(filter_diff->data_type()));
      return;
    }

    const int& groups = ctx->Attr<int32_t>("groups");
    std::vector<int32_t> paddings = ctx->Attr<std::vector<int32_t>>("padding_before");
    std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    std::vector<int32_t> dilation_rates = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const std::string& data_format = ctx->Attr<std::string>("data_format");

    auto dy_shape = Shape(dy->shape_view());
    auto x_shape = Shape(x->shape_view());
    auto filter_diff_shape = Shape(filter_diff->shape_view());
    const void* dy_ptr = dy->dptr();
    const void* x_ptr = x->dptr();
    void* filter_diff_ptr = filter_diff->mut_dptr();

    int32_t kernel_dims = x_shape.NumAxes() - 2;
    UpdateConvParams(&paddings, &strides, &dilation_rates, kernel_dims);

    auto data_type = dy->data_type();
    auto cnnl_data_type = ConvertToCnnlDataType(data_type);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDescriptor dy_desc, x_desc, filter_diff_desc;
    CnnlConvolutionDescriptor conv_desc;
    conv_desc.set(x_shape.NumAxes(), strides.data(), paddings.data(), dilation_rates.data(), groups,
                  cnnl_data_type);

    CnnlWorkspace temp_dy(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_x(ctx->stream()->As<ep::MluStream>());
    CnnlWorkspace temp_filter_diff(ctx->stream()->As<ep::MluStream>());

    if (data_format != "channels_last") {
      size_t element_size = GetSizeOfDataType(data_type);
      dy_shape = mlu::ComputeShapeNchwToNhwc(dy_shape);
      x_shape = mlu::ComputeShapeNchwToNhwc(x_shape);
      filter_diff_shape = mlu::ComputeShapeNchwToNhwc(filter_diff_shape);
      temp_dy.resize(dy_shape.elem_cnt() * element_size);
      temp_x.resize(x_shape.elem_cnt() * element_size);
      temp_filter_diff.resize(filter_diff_shape.elem_cnt() * element_size);
      // convert dy to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), dy->shape_view(), data_type, dy->dptr(),
                               temp_dy.dptr(), MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      // convert x to NHWC
      mlu::ConvertMemoryFormat(ctx->stream(), x->shape_view(), data_type, x->dptr(), temp_x.dptr(),
                               MemoryFormat::kNCHW, MemoryFormat::kNHWC);
      dy_ptr = temp_dy.dptr();
      x_ptr = temp_x.dptr();
      filter_diff_ptr = temp_filter_diff.dptr();
    }
    dy_desc.set(dy_shape.NumAxes(), dy_shape.data(), cnnl_data_type, layout);
    x_desc.set(x_shape.NumAxes(), x_shape.data(), cnnl_data_type, layout);
    filter_diff_desc.set(filter_diff_shape.NumAxes(), filter_diff_shape.data(), cnnl_data_type,
                         layout);

    cnnlConvolutionBwdFilterAlgo_t algo;
    OF_CNNL_CHECK(cnnlGetConvolutionBackwardFilterAlgorithm(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), conv_desc.desc(), x_desc.desc(),
        dy_desc.desc(), filter_diff_desc.desc(), CNNL_CONVOLUTION_BWD_FILTER_FASTEST, &algo));

    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetConvolutionBackwardFilterWorkspaceSize(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), x_desc.desc(), dy_desc.desc(),
        filter_diff_desc.desc(), conv_desc.desc(), algo, &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);

    OF_CNNL_CHECK(cnnlConvolutionBackwardFilter(
        ctx->stream()->As<ep::MluStream>()->cnnl_handle(), nullptr, x_desc.desc(), x_ptr,
        dy_desc.desc(), dy_ptr, conv_desc.desc(), algo, workspace.dptr(), workspace_size, nullptr,
        filter_diff_desc.desc(), filter_diff_ptr));

    if (data_format != "channels_last") {
      // convert filter_diff to NCHW
      mlu::ConvertMemoryFormat(ctx->stream(), filter_diff_shape, data_type, filter_diff_ptr,
                               filter_diff->mut_dptr(), MemoryFormat::kNHWC, MemoryFormat::kNCHW);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONV_FILTER_GRAD_MLU_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("conv_filter_grad")                            \
      .SetCreateFn<ConvFilterGradKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CONV_FILTER_GRAD_MLU_KERNEL(float)
REGISTER_CONV_FILTER_GRAD_MLU_KERNEL(float16)

}  // namespace oneflow
