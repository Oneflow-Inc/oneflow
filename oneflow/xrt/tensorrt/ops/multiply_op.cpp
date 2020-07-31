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
#include "absl/strings/str_cat.h"
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

class MultiplyOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape x_shape = ctx->InputShape("x_0");
    Shape y_shape = ctx->InputShape("y_0");
    nvinfer1::ITensor *x = ctx->Input("x_0");
    nvinfer1::ITensor *y = ctx->Input("y_0");
    CHECK_EQ(x_shape, y_shape);
    auto *layer = ctx->builder()->addElementWise(*x, *y, nvinfer1::ElementWiseOperation::kPROD); 
    ctx->SetSoleOutput(layer->getOutput(0));  
  }
};

REGISTER_TRT_OP_KERNEL(Multiply, MultiplyOp)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
