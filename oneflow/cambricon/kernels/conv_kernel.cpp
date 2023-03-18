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
#include "cnnl.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"

namespace oneflow {

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx, const int& num_dims) {
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

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

std::vector<int32_t> ComputePermutation(int32_t ndim, cnnlTensorLayout_t layout) {
  CHECK_GT(ndim, 2);
  CHECK(layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NCHW);
  std::vector<int32_t> permute(ndim);
  if (layout == CNNL_LAYOUT_NHWC) {
    // NCHW -> NHWC
    permute[0] = 0;
    permute[ndim - 1] = 1;
    for (int i = 0; i < ndim - 2; ++i) { permute[i + 1] = i + 2; }
  } else {
    // NHWC -> NCHW
    permute[0] = 0;
    permute[1] = ndim - 1;
    for (int i = 0; i < ndim - 2; ++i) { permute[i + 2] = i + 1; }
  }
  return permute;
}

std::vector<int64_t> ComputePermuteShape(const ShapeView& shape,
                                         const std::vector<int32_t>& permute) {
  CHECK_EQ(shape.NumAxes(), permute.size());
  std::vector<int64_t> permute_shape(shape.NumAxes());
  for (int i = 0; i < permute.size(); ++i) { permute_shape[i] = shape[permute[i]]; }
  return permute_shape;
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

    const auto& in_shape = in->shape_view();
    const auto& weight_shape = weight->shape_view();
    const auto& out_shape = out->shape_view();

    int32_t kernel_size = in_shape.NumAxes() - 2;
    UpdateConvParams(&paddings, &strides, &dilation_rates, kernel_size);

    auto cnnl_data_type = ConvertToCnnlDataType(in->data_type());
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDescriptor input_desc, weight_desc, bias_desc, output_desc;
    CnnlConvolutionDescriptor conv_desc;
    conv_desc.set(in_shape.NumAxes(), strides.data(), paddings.data(), dilation_rates.data(),
                  groups, cnnl_data_type);

    const void* input_ptr = in->dptr();
    const void* weight_ptr = weight->dptr();
    void* output_ptr = out->mut_dptr();

    CnnlWorkspace temp_input(ctx->stream()->As<ep::MluStream>(), 0);
    CnnlWorkspace temp_weight(ctx->stream()->As<ep::MluStream>(), 0);
    CnnlWorkspace temp_output(ctx->stream()->As<ep::MluStream>(), 0);

    size_t element_size = GetSizeOfDataType(in->data_type());
    std::vector<int64_t> temp_output_shape;

    if (data_format != "channels_last") {
      auto permute = ComputePermutation(out_shape.NumAxes(), layout);
      temp_input.resize(in_shape.elem_cnt() * element_size);
      auto transpose = NewPermutePrimitive(ctx, in_shape.NumAxes());
      CHECK(transpose);
      // transpose input from NCHW to NHWC
      transpose->Launch(ctx->stream(), in->data_type(), in_shape.NumAxes(), in_shape.data(),
                        in->dptr(), permute.data(), temp_input.dptr());
      input_desc.set(in_shape.NumAxes(), ComputePermuteShape(in_shape, permute).data(),
                     cnnl_data_type, layout);
      input_ptr = temp_input.dptr();

      temp_output_shape = ComputePermuteShape(out_shape, permute);
      output_desc.set(out_shape.NumAxes(), temp_output_shape.data(), cnnl_data_type, layout);
      temp_output.resize(out_shape.elem_cnt() * element_size);
      output_ptr = temp_output.dptr();
    } else {
      input_desc.set(in_shape.NumAxes(), in_shape.data(), cnnl_data_type, layout);
      weight_desc.set(weight_shape.NumAxes(), weight_shape.data(), cnnl_data_type, layout);
      output_desc.set(out_shape.NumAxes(), out_shape.data(), cnnl_data_type, layout);
    }

    auto transpose = NewPermutePrimitive(ctx, weight_shape.NumAxes());
    CHECK(transpose);
    temp_weight.resize(weight_shape.elem_cnt() * element_size);
    auto permute = ComputePermutation(weight_shape.NumAxes(), layout);
    transpose->Launch(ctx->stream(), weight->data_type(), weight_shape.NumAxes(),
                      weight_shape.data(), weight->dptr(), permute.data(), temp_weight.dptr());
    weight_desc.set(weight_shape.NumAxes(), ComputePermuteShape(weight_shape, permute).data(),
                    cnnl_data_type, layout);
    weight_ptr = temp_weight.dptr();

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
      auto transpose = NewPermutePrimitive(ctx, out_shape.NumAxes());
      CHECK(transpose);
      // transpose output from NHWC to NCHW
      auto permute = ComputePermutation(out_shape.NumAxes(), CNNL_LAYOUT_NCHW);
      transpose->Launch(ctx->stream(), out->data_type(), out_shape.NumAxes(),
                        temp_output_shape.data(), output_ptr, permute.data(), out->mut_dptr());
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

}  // namespace oneflow
