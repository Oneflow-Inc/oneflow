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
#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::UnaryOperation unary_op>
class UnaryOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* x = ctx->SoleInput();
    auto* layer = ctx->builder()->addUnary(*x, unary_op);
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Sqrt, UnaryOp<nvinfer1::UnaryOperation::kSQRT>)
    .EnableTrainPhase()
    .Finalize();

class RsqrtOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* x = ctx->SoleInput();
    auto* sqrt_layer = ctx->builder()->addUnary(*x, nvinfer1::UnaryOperation::kSQRT);
    auto* layer = ctx->builder()->addUnary(*(sqrt_layer->getOutput(0)), nvinfer1::UnaryOperation::kRECIP);
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(Rsqrt, RsqrtOp).EnableTrainPhase().Finalize();

class SqrtGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    nvinfer1::ITensor* x = ctx->Input("x_0");
    nvinfer1::ITensor* dy = ctx->Input("dy_0");
    // x^0.5
    auto* sqrt_layer = ctx->builder()->addUnary(*x, nvinfer1::UnaryOperation::kSQRT);

    // 2 * x^0.5
    const Shape& x_shape = ctx->InputShape("x_0");
    Shape shape(DimVector(x_shape.NumAxes(), 1));

    DataType data_type = ctx->InputType("x_0");
    std::string name = ctx->op_name() + "_2";
    nvinfer1::Weights constant = helpers::Constant(ctx, 2, shape, data_type, name);
    auto* constant_layer = ctx->builder()->addConstant(ShapeToXrtDims(shape), constant);
    auto* mul_layer = ctx->builder()->addElementWise(*(sqrt_layer->getOutput(0)), *(constant_layer->getOutput(0)), nvinfer1::ElementWiseOperation::kPROD);

    // 1 / (2 * x^0.5)
    auto* layer = ctx->builder()->addUnary(*(mul_layer->getOutput(0)), nvinfer1::UnaryOperation::kRECIP);
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(SqrtGrad, SqrtGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
