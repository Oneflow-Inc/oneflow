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
// #include "oneflow/xrt/tensorrt/plugin/prelu_grad_plugin.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class PReluGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    // TODO(hjchen2): plugin maybe faster but we have not implement it.
    // std::vector<nvinfer1::ITensor*> inputs(3);
    // inputs[0] = ctx->Input("dy_0");
    // inputs[1] = ctx->Input("x_0");
    // inputs[2] = ctx->Input("alpha_0");
    // bool alpha_requires_grad = ctx->Attr<bool>("alpha_requires_grad");
    // PreluGradPlugin plugin(ctx->op_name(), alpha_requires_grad);
    // auto* layer = ctx->builder()->addPluginV2(inputs.data(), 3, plugin);

    nvinfer1::ITensor* dy = ctx->Input("dy_0");
    nvinfer1::ITensor* x = ctx->Input("x_0");
    nvinfer1::ITensor* alpha = ctx->Input("alpha_0");

    // dx = dy * (x > 0 ? 1 : alpha), equals to
    // dx = dy * (x > 0 ? x : alpha * x) / x, equals to
    // dx = dy * PRelu(x) / x
    const Shape& x_shape = ctx->InputShape("x_0");
    const Shape& alpha_shape = ctx->InputShape("alpha_0");
    CHECK_GE(x_shape.NumAxes(), 2) << "input rank should >= 2";
    CHECK_EQ(alpha_shape.NumAxes(), 1) << "alpha rank should be 1";
    int64_t channels = x_shape.At(1);
    CHECK_EQ(alpha_shape.elem_cnt(), channels) << "alpha element count should be equal to channels";
    DimVector shape(x_shape.NumAxes(), 1);
    shape[1] = channels;
    alpha = helpers::Reshape(ctx, alpha, Shape(shape));

    auto* prelu = ctx->builder()->addParametricReLU(*x, *alpha);
    std::string prelu_name = ctx->op_name() + ".prelu";
    prelu->setName(prelu_name.c_str());

    auto* div = ctx->builder()->addElementWise(*(prelu->getOutput(0)), *x,
                                               nvinfer1::ElementWiseOperation::kDIV);
    std::string div_name = ctx->op_name() + ".div";
    div->setName(div_name.c_str());

    auto* mul = ctx->builder()->addElementWise(*dy, *(div->getOutput(0)),
                                               nvinfer1::ElementWiseOperation::kPROD);
    std::string mul_name = ctx->op_name() + ".mul";
    mul->setName(mul_name.c_str());

    ctx->SetOutput("dx_0", mul->getOutput(0));

    bool alpha_requires_grad = ctx->Attr<bool>("alpha_requires_grad");
    if (alpha_requires_grad) {
      // da = reduce(x > 0 ? 0 : dy * x), equals to
      // da = reduce(dy * (x > 0 ? 0 : x)), equals to
      // da = reduce(dy * (- (-x > 0 ? -x : 0))), equals to
      // da = reduce(dy * (-Relu(-x)))
      auto* neg0 = ctx->builder()->addUnary(*x, nvinfer1::UnaryOperation::kNEG);
      std::string neg0_name = ctx->op_name() + ".neg0";
      neg0->setName(neg0_name.c_str());

      auto* relu =
          ctx->builder()->addActivation(*(neg0->getOutput(0)), nvinfer1::ActivationType::kRELU);
      std::string relu_name = ctx->op_name() + ".relu";
      relu->setName(relu_name.c_str());

      auto* neg1 = ctx->builder()->addUnary(*(relu->getOutput(0)), nvinfer1::UnaryOperation::kNEG);
      std::string neg1_name = ctx->op_name() + ".neg1";
      neg1->setName(neg1_name.c_str());

      uint32_t reduce_axes = 0;
      for (int i = 0; i < x_shape.NumAxes() - 1; ++i) {
        if (i != 1) { reduce_axes |= (1U << i); }
      }
      auto* reduce = ctx->builder()->addReduce(
          *(neg1->getOutput(0)), nvinfer1::ReduceOperation::kSUM, reduce_axes, /*keepdims*/ false);
      ctx->SetOutput("alpha_diff_0", reduce->getOutput(0));
    }
  }
};

REGISTER_TRT_OP_KERNEL(PReluGrad, PReluGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
