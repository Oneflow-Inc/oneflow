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

template<nvinfer1::ReduceOperation reduce_op>
class ReduceOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    const auto &axis = ctx->Attr<std::vector<int32_t>>("axis");

    int32_t reduce_axis = 0;
    for (int i = 0; i < axis.size(); ++i) { reduce_axis = reduce_axis | (1U << axis[i]); }
    bool keepDimensions = ctx->Attr<bool>("keepdims");
    // TensorRT does not support full reduce without keepDimensions.
    Shape in_shape = ctx->SoleInputShape();
    if (!keepDimensions) {
      CHECK_NE(reduce_axis, (1U << in_shape.NumAxes()) - 1)
          << "TensorRT does not support full reduce without keepDimensions.";
    }

    nvinfer1::ITensor *in = ctx->SoleInput();
    auto *layer = ctx->builder()->addReduce(*in, reduce_op, reduce_axis, keepDimensions);
    layer->setName(ctx->op_name().c_str());
    ctx->SetSoleOutput(layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(ReduceSum, ReduceOp<nvinfer1::ReduceOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();
REGISTER_TRT_OP_KERNEL(ReduceMean, ReduceOp<nvinfer1::ReduceOperation::kAVG>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
