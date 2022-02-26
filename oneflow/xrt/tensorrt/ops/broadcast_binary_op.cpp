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
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"
#include "oneflow/xrt/tensorrt/trt_helpers.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

template<nvinfer1::ElementWiseOperation element_wise_op>
class BcastBinaryOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape shape_a = ctx->InputShape("x_0");
    Shape shape_b = ctx->InputShape("y_0");

    int axes = std::max(shape_a.NumAxes(), shape_b.NumAxes());
    shape_a = CreateLeftExtendedShape(ShapeView(shape_a), axes);
    shape_b = CreateLeftExtendedShape(ShapeView(shape_b), axes);

    nvinfer1::ITensor* x = helpers::Reshape(ctx, ctx->Input("x_0"), shape_a);
    nvinfer1::ITensor* y = helpers::Reshape(ctx, ctx->Input("y_0"), shape_b);
    auto* layer = ctx->builder()->addElementWise(*x, *y, element_wise_op);
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(BcastAdd, BcastBinaryOp<nvinfer1::ElementWiseOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(BcastSub, BcastBinaryOp<nvinfer1::ElementWiseOperation::kSUB>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(BcastMul, BcastBinaryOp<nvinfer1::ElementWiseOperation::kPROD>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(BcastDiv, BcastBinaryOp<nvinfer1::ElementWiseOperation::kDIV>)
    .EnableTrainPhase()
    .Finalize();


}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
