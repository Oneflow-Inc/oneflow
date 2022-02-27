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

namespace oneflow {
namespace xrt {
namespace tensorrt {

class ConvDataGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* input = ctx->Input("dy_0");
    nvinfer1::Weights weight = ctx->Weight("filter_0");

    CHECK_EQ(ctx->Attr<std::string>("data_format"), "channels_first");
    const auto& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const int groups = ctx->Attr<int32_t>("groups");

    // int num_spatial_dims = ctx->Attr<int32_t>("num_spatial_dims");
    const Shape& weight_shape = ctx->InputShape("filter_0");
    int32_t filters = weight_shape.At(1);

    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT /* type */, nullptr /* values */,
                           0 /* count */};
    auto* layer = ctx->builder()->addDeconvolution(
        *input, filters, nvinfer1::DimsHW(kernel_size[0], kernel_size[1]), weight, bias);
    layer->setName(ctx->op_name().c_str());

    const auto& strides = ctx->Attr<std::vector<int32_t>>("strides");
    const auto& dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");

    layer->setStride(nvinfer1::DimsHW(strides[0], strides[1]));
    layer->setDilationNd(nvinfer1::DimsHW(dilation[0], dilation[1]));
    layer->setNbGroups(groups);

    auto pads = ctx->Attr<std::vector<int32_t>>("padding_before");
    const auto& dy_shape = ctx->InputShape("dy_0");
    const auto& like_shape = ctx->InputShape("x_like_0");
    std::vector<int32_t> output_pads(pads.size(), 0);
    bool need_output_pad = false;
    for (int i = 0; i < pads.size(); ++i) {
      int32_t output_size = (dy_shape.At(2 + i) - 1) * strides[i] - 2 * pads[i]
                            + dilation[i] * (kernel_size[i] - 1) + 1;
      pads[i] -= (like_shape.At(2 + i) - output_size + 1) / 2;
      if (pads[i] < 0) {
        output_pads[i] = -pads[i];
        pads[i] = 0;
        need_output_pad = true;
      }
    }
    layer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
    layer->setPrePadding(nvinfer1::DimsHW(pads[0], pads[1]));
    layer->setPostPadding(nvinfer1::DimsHW(pads[0], pads[1]));

    if (need_output_pad) {
      auto* pad_layer = ctx->builder()->addPaddingNd(
          *(layer->getOutput(0)), nvinfer1::DimsHW(output_pads[0], output_pads[1]),
          nvinfer1::DimsHW(0, 0));
      std::string name = ctx->op_name() + "_output_padding";
      pad_layer->setName(name.c_str());
      ctx->SetOutput("dx_0", pad_layer->getOutput(0));
    } else {
      ctx->SetOutput("dx_0", layer->getOutput(0));
    }
  }
};

// REGISTER_TRT_OP_KERNEL(ConvDataGrad, ConvDataGradOp).Finalize();
REGISTER_TRT_OP_KERNEL(ConvDataGrad, ConvDataGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
