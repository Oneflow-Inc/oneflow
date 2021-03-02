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

#include "oneflow/xrt/api.h"
#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class BiasAddOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    CHECK_EQ(ctx->InputType("a_0"), ctx->InputType("b_0"));

    Shape in_shape = ctx->InputShape("a_0");
    Shape bias_shape = ctx->InputShape("b_0");
    CHECK_GE(in_shape.NumAxes(), 2);
    CHECK_EQ(bias_shape.NumAxes(), 1);

    std::vector<int64_t> dims(in_shape.NumAxes(), 1);
    int32_t axis = ctx->Attr<int32_t>("axis");
    dims[axis] = bias_shape.At(0);

    nvinfer1::ITensor *in = ctx->Input("a_0");;
    nvinfer1::Weights bias = ctx->Weight("b_0");
    nvinfer1::ITensor *reshaped_bias = helpers::Reshape(ctx, bias, AsShape(dims));
    // Add bias to input by ElementWise layer.
    auto *layer = ctx->builder()->addElementWise(  // NOLINT
        *in, *reshaped_bias, nvinfer1::ElementWiseOperation::kSUM);
    layer->setName(ctx->op_name().c_str());

    ctx->SetOutput("out_0", layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(BiasAdd, BiasAddOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
