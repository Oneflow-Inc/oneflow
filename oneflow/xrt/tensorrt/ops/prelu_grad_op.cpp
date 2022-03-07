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
    // dx = dy * (alpha + (1 - alpha) * (x > 0 ? 1 : 0)), equals to
    // dx = dy * (alpha + (1 - alpha) * ceil(clip(x, 0, 1)))
    const Shape& x_shape = ctx->InputShape("x_0");
    const Shape& alpha_shape = ctx->InputShape("alpha_0");
    CHECK_GE(x_shape.NumAxes(), 2) << "input rank should >= 2";
    CHECK_EQ(alpha_shape.NumAxes(), 1) << "alpha rank should be 1";
    int64_t channels = x_shape.At(1);
    CHECK_EQ(alpha_shape.elem_cnt(), channels) << "alpha element count should be equal to channels";
    Shape shape(DimVector(x_shape.NumAxes(), 1));
    shape.Set(1, channels);
    alpha = helpers::Reshape(ctx, alpha, shape);

    DataType data_type = ctx->InputType("alpha_0");
    std::string ones_name = ctx->op_name() + ".ones";
    auto* ones = ctx->builder()->addConstant(
        ShapeToXrtDims(shape), helpers::Constant(ctx, 1.f, shape, data_type, ones_name));
    ones->setName(ones_name.c_str());
    // 1 - alpha
    auto* sub_alpha = ctx->builder()->addElementWise(*(ones->getOutput(0)), *alpha,
                                                     nvinfer1::ElementWiseOperation::kSUB);
    std::string sub_alpha_name = ctx->op_name() + ".sub_alpha";
    sub_alpha->setName(sub_alpha_name.c_str());
    // clip(x, 0, 1)
    auto* clip = ctx->builder()->addActivation(*x, nvinfer1::ActivationType::kCLIP);
    clip->setAlpha(0.f);
    clip->setBeta(1.f);
    std::string clip_name = ctx->op_name() + ".clip";
    clip->setName(clip_name.c_str());
    // ceil(clip(x, 0, 1))
    auto* ceil = ctx->builder()->addUnary(*(clip->getOutput(0)), nvinfer1::UnaryOperation::kCEIL);
    std::string ceil_name = ctx->op_name() + ".ceil";
    ceil->setName(ceil_name.c_str());
    // (1 - alpha) * ceil(clip(x, 0, 1))
    auto* mul0 = ctx->builder()->addElementWise(*(sub_alpha->getOutput(0)), *(ceil->getOutput(0)),
                                                nvinfer1::ElementWiseOperation::kPROD);
    std::string mul0_name = ctx->op_name() + ".mul0";
    mul0->setName(mul0_name.c_str());
    // alpha + (1 - alpha) * ceil(clip(x, 0, 1))
    auto* add = ctx->builder()->addElementWise(*alpha, *(mul0->getOutput(0)),
                                               nvinfer1::ElementWiseOperation::kSUM);
    std::string add_name = ctx->op_name() + ".add";
    add->setName(add_name.c_str());
    // dy * (alpha + (1 - alpha) * ceil(clip(x, 0, 1)))
    auto* mul1 = ctx->builder()->addElementWise(*dy, *(add->getOutput(0)),
                                                nvinfer1::ElementWiseOperation::kPROD);
    std::string mul1_name = ctx->op_name() + ".mul1";
    mul1->setName(mul1_name.c_str());

    ctx->SetOutput("dx_0", mul1->getOutput(0));

    bool alpha_requires_grad = ctx->Attr<bool>("alpha_requires_grad");
    if (alpha_requires_grad) {
      // da = reduce(x > 0 ? 0 : dy * x), equals to
      // da = reduce(dy * x * (x > 0 ? 0 : 1)), equals to
      // da = reduce(dy * x * (1 - (x > 0 ? 1 : 0))), equals to
      // da = reduce(dy * x * (1 - ceil(clip(x, 0, 1))))

      auto* sub = ctx->builder()->addElementWise(*(ones->getOutput(0)), *(ceil->getOutput(0)),
                                                 nvinfer1::ElementWiseOperation::kSUB);
      std::string sub_name = ctx->op_name() + ".alpha_diff.sub";
      sub->setName(sub_name.c_str());
      auto* mul2 = ctx->builder()->addElementWise(*x, *(sub->getOutput(0)),
                                                  nvinfer1::ElementWiseOperation::kPROD);
      std::string mul2_name = ctx->op_name() + ".alpha_diff.mul2";
      mul2->setName(mul2_name.c_str());
      auto* mul3 = ctx->builder()->addElementWise(*dy, *(mul2->getOutput(0)),
                                                  nvinfer1::ElementWiseOperation::kPROD);
      std::string mul3_name = ctx->op_name() + ".alpha_diff.mul3";
      mul3->setName(mul3_name.c_str());

      uint32_t reduce_axes = 0;
      for (int i = 0; i < x_shape.NumAxes() - 1; ++i) {
        if (i != 1) { reduce_axes |= (1U << i); }
      }
      auto* reduce = ctx->builder()->addReduce(
          *(mul3->getOutput(0)), nvinfer1::ReduceOperation::kSUM, reduce_axes, /*keepdims*/ false);
      std::string reduce_name = ctx->op_name() + ".alpha_diff.reduce";
      reduce->setName(reduce_name.c_str());
      ctx->SetOutput("alpha_diff_0", reduce->getOutput(0));
    }
  }
};

REGISTER_TRT_OP_KERNEL(PReluGrad, PReluGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
