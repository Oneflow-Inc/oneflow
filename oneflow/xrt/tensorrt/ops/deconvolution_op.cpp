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
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class DeConvolutionOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* input = ctx->Input("in_0");
    nvinfer1::Weights weight = ctx->Weight("weight_0");

    CHECK_EQ(ctx->Attr<std::string>("data_format"), "channels_first");
    const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    int filters = ctx->Attr<int32_t>("filters");

    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT /* type */, nullptr /* values */,
                           0 /* count */};
    auto* layer = ctx->builder()->addDeconvolution(
        *input, filters, nvinfer1::DimsHW(kernel_size[0], kernel_size[1]), weight, bias);
    layer->setName(ctx->op_name().c_str());

    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    const int groups = ctx->Attr<int32_t>("groups");

    layer->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
    layer->setDilationNd(nvinfer1::DimsHW(dilation[0], dilation[1]));
    layer->setNbGroups(groups);

    const auto& paddings = ctx->Attr<std::vector<int32_t>>("padding_before");
    layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
    layer->setPrePadding(nvinfer1::DimsHW(paddings[0], paddings[1]));
    layer->setPostPadding(nvinfer1::DimsHW(paddings[0], paddings[1]));

    const auto& output_padding = ctx->Attr<std::vector<int32_t>>("output_padding");
    if (output_padding.size() == 2 && output_padding[0] && output_padding[1]) {
      auto* pad_layer = ctx->builder()->addPaddingNd(
          *(layer->getOutput(0)), nvinfer1::DimsHW(output_padding[0], output_padding[1]),
          nvinfer1::DimsHW(0, 0));
      std::string name = ctx->op_name() + "_output_padding";
      pad_layer->setName(name.c_str());
      ctx->SetOutput("out_0", pad_layer->getOutput(0));
    } else {
      ctx->SetOutput("out_0", layer->getOutput(0));
    }
  }
};

REGISTER_TRT_OP_KERNEL(DeConv2D, DeConvolutionOp).Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
