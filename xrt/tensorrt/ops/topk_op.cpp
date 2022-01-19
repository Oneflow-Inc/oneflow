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

class TopKOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    // CHECK_GE(in_shape.NumAxes(), 2);

    int32_t k = ctx->Attr<int32_t>("k");
    // We only compute the top-k at the last dimension.
    // CHECK_GE(axis, 1);
    // CHECK_LT(axis, in_shape.NumAxes());
    uint32_t reduce_axis = (1U << (in_shape.NumAxes() - 1));
    nvinfer1::ITensor* in = ctx->SoleInput();
    auto* layer = ctx->builder()->addTopK(*in, nvinfer1::TopKOperation::kMAX, k, reduce_axis);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(TopK, TopKOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
