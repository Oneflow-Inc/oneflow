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
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::ElementWiseOperation element_wise_op>
class ScalarBinaryOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    const Shape& in_shape = ctx->SoleInputShape();
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = ctx->Attr<int64_t>("int_operand");
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = ctx->Attr<double>("float_operand");
    }    
    DataType data_type = ctx->SoleInputType();
    Shape shape(DimVector(in_shape.NumAxes(), 1));
    std::string name = ctx->op_name() + "_scalar";
    nvinfer1::Weights constant = helpers::Constant(ctx, value, shape, data_type, name);
    auto* constant_layer = ctx->builder()->addConstant(ShapeToXrtDims(shape), constant);

    nvinfer1::ITensor* scalar = constant_layer->getOutput(0);
    nvinfer1::ITensor* in = ctx->SoleInput();
    auto* layer = ctx->builder()->addElementWise(*in, *scalar, element_wise_op);
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(ScalarAdd, ScalarBinaryOp<nvinfer1::ElementWiseOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(ScalarSub, ScalarBinaryOp<nvinfer1::ElementWiseOperation::kSUB>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(ScalarMul, ScalarBinaryOp<nvinfer1::ElementWiseOperation::kPROD>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(ScalarDiv, ScalarBinaryOp<nvinfer1::ElementWiseOperation::kDIV>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(ScalarPow, ScalarBinaryOp<nvinfer1::ElementWiseOperation::kPOW>)
    .EnableTrainPhase()
    .Finalize();


class ScalarPowGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    // dy * scalar * x^(scalar - 1)
    const Shape& in_shape = ctx->InputShape("x_0");
    Scalar value;
    if (ctx->Attr<bool>("has_int_operand")) {
      value = ctx->Attr<int64_t>("int_operand");
    } else if (ctx->Attr<bool>("has_float_operand")) {
      value = ctx->Attr<double>("float_operand");
    }
    DataType data_type = ctx->InputType("x_0");
    Shape shape(DimVector(in_shape.NumAxes(), 1));
    nvinfer1::ITensor* scalar = nullptr;
    nvinfer1::ITensor* scalar_sub_1 = nullptr;
    {
      std::string name = ctx->op_name() + "_scalar";
      nvinfer1::Weights constant = helpers::Constant(ctx, value, shape, data_type, name);
      auto* constant_layer = ctx->builder()->addConstant(ShapeToXrtDims(shape), constant);
      scalar = constant_layer->getOutput(0);
    }
    {
      std::string name = ctx->op_name() + "_scalar_sub_1";
      nvinfer1::Weights constant = helpers::Constant(ctx, value - 1, shape, data_type, name);
      auto* constant_layer = ctx->builder()->addConstant(ShapeToXrtDims(shape), constant);
      scalar_sub_1 = constant_layer->getOutput(0);
    }
    // x^(scalar - 1)
    auto* pow_layer = ctx->builder()->addElementWise(*(ctx->Input("x_0")), *scalar_sub_1, nvinfer1::ElementWiseOperation::kPOW);
    // scalar * x^(scalar - 1)
    auto* mul_layer = ctx->builder()->addElementWise(*(pow_layer->getOutput(0)), *scalar, nvinfer1::ElementWiseOperation::kPROD);
    // dy * scalar * x^(scalar - 1)
    auto* layer = ctx->builder()->addElementWise(*(mul_layer->getOutput(0)), *(ctx->Input("dy_0")), nvinfer1::ElementWiseOperation::kPROD);
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(ScalarPowGrad, ScalarPowGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
