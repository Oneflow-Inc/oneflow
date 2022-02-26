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
#include "NvInferRuntimeCommon.h"  // nvinfer1::ActivationType

#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::ActivationType activation_type>
class ActivationOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* in = ctx->SoleInput();
    auto* layer = ctx->builder()->addActivation(*in, activation_type);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Tanh, ActivationOp<nvinfer1::ActivationType::kTANH>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(Relu, ActivationOp<nvinfer1::ActivationType::kRELU>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(Sigmoid, ActivationOp<nvinfer1::ActivationType::kSIGMOID>)
    .EnableTrainPhase()
    .Finalize();

template<>
class ActivationOp<nvinfer1::ActivationType::kLEAKY_RELU> : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* in = ctx->SoleInput();
    auto* layer = ctx->builder()->addActivation(*in, nvinfer1::ActivationType::kLEAKY_RELU);
    layer->setAlpha(ctx->Attr<float>("alpha"));
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(LeakyRelu, ActivationOp<nvinfer1::ActivationType::kLEAKY_RELU>)
    .EnableTrainPhase()
    .Finalize();

class PReluOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* x = ctx->Input("x_0");
    nvinfer1::ITensor* alpha = ctx->Input("alpha_0");

    Shape x_shape = ctx->InputShape("x_0");
    Shape alpha_shape = ctx->InputShape("alpha_0");
    if (alpha_shape.NumAxes() != x_shape.NumAxes()) {
      CHECK_EQ(alpha_shape.NumAxes(), 1) << "alpha rank should be 1";
      int64_t channels = 1;
      if (x_shape.NumAxes() > 1) {
        channels = x_shape.At(1);
      }
      CHECK_EQ(alpha_shape.elem_cnt(), channels) << "alpha element count should be equal to channels";
      DimVector shape(x_shape.NumAxes(), 1);
      shape[1] = channels;
      alpha = helpers::Reshape(ctx, alpha, Shape(shape));
    }
    auto* layer = ctx->builder()->addParametricReLU(*x, *alpha);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(PRelu, PReluOp).EnableTrainPhase().Finalize();


}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
