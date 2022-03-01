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

// dy = reduce(-dz * (z / y))
class BroadcastDivGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    // r0 = z / y
    auto* r0 = ctx->builder()->addElementWise(*(ctx->Input("z_0")), *(ctx->Input("y_0")),
                                              nvinfer1::ElementWiseOperation::kDIV);
    // r1 = dz * r0
    auto* r1 = ctx->builder()->addElementWise(*(ctx->Input("dz_0")), *(r0->getOutput(0)),
                                              nvinfer1::ElementWiseOperation::kPROD);
    // r3 = -r1
    auto* r3 = ctx->builder()->addUnary(*(r1->getOutput(0)), nvinfer1::UnaryOperation::kNEG);
    // reduce
    const auto& large_shape = ctx->InputShape("z_0");
    const auto& reduced_shape = ctx->InputShape("y_0");
    uint32_t reduce_axes = 0;
    for (int i = large_shape.NumAxes() - 1; i >= reduced_shape.NumAxes(); --i) {
      reduce_axes |= (1U << i);
    }
    for (int i = reduced_shape.NumAxes() - 1; i >= 0; --i) {
      if (large_shape.At(i) != reduced_shape.At(i)) { reduce_axes |= (1U << i); }
    }
    auto* reduce = ctx->builder()->addReduce(*(r3->getOutput(0)), nvinfer1::ReduceOperation::kSUM,
                                             reduce_axes, /*keepdims*/ true);
    ctx->SetOutput("dy_0", helpers::Reshape(ctx, reduce->getOutput(0), reduced_shape));
  }
};

REGISTER_TRT_OP_KERNEL(BcastDivGrad, BroadcastDivGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
