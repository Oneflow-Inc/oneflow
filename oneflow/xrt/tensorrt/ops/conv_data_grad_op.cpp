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
#include "NvInfer.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ConvDataGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* input = ctx->Input("dy_0");
    nvinfer1::Weights weight = ctx->Weight("filter_0");
    const auto& dy_shape = ctx->InputShape("dy_0");
    const auto& x_shape = ctx->InputShape("x_like_0");

    CHECK_EQ(ctx->Attr<std::string>("data_format"), "channels_first");
    const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const auto& paddings = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const int groups = ctx->Attr<int32_t>("groups");

    const Shape& weight_shape = ctx->InputShape("filter_0");
    int32_t filters = weight_shape.At(1);
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT /* type */, nullptr /* values */,
                           0 /* count */};
    auto* deconv = ctx->builder()->addDeconvolution(
        *input, filters, nvinfer1::DimsHW(kernel_size[0], kernel_size[1]), weight, bias);
    deconv->setName(ctx->op_name().c_str());

    deconv->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
    deconv->setPrePadding(nvinfer1::DimsHW(paddings[0], paddings[1]));
    deconv->setPostPadding(nvinfer1::DimsHW(0, 0));
    deconv->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
    deconv->setDilationNd(nvinfer1::DimsHW(dilation[0], dilation[1]));
    deconv->setNbGroups(groups);

    nvinfer1::ITensor* output = deconv->getOutput(0);
    nvinfer1::Dims start, size, stride;
    start.nbDims = x_shape.NumAxes();
    size.nbDims = start.nbDims;
    stride.nbDims = start.nbDims;
    for (int i = 0; i < start.nbDims; ++i) {
      start.d[i] = 0;
      size.d[i] = x_shape.At(i);
      stride.d[i] = 1;
    }
    if (!helpers::DimsEqual(size, output->getDimensions())) {
      auto* slice_output = ctx->builder()->addSlice(*output, start, size, stride);
      slice_output->setMode(nvinfer1::SliceMode::kFILL);
      std::string name = ctx->op_name() + ".slice_output";
      slice_output->setName(name.c_str());

      // add identity layer after slice to bypass some internal error,
      // refer to https://github.com/NVIDIA/TensorRT/issues/1821
      auto* identity = ctx->builder()->addIdentity(*(slice_output->getOutput(0)));
      std::string identity_name = ctx->op_name() + ".identity";
      identity->setName(identity_name.c_str());
      output = identity->getOutput(0);
    }
    ctx->SetOutput("dx_0", output);
  }
};

REGISTER_TRT_OP_KERNEL(ConvDataGrad, ConvDataGradOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
