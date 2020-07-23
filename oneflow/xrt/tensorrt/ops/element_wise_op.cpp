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

template<nvinfer1::ElementWiseOperation element_wise_op>
class ElementWiseOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    int num_inputs = ctx->num_inputs();
    CHECK_GE(num_inputs, 2) << "ElementWiseOp needs 2 inputs at least.";

    Shape in_shape = ctx->InputShape("in_0");
    nvinfer1::ITensor *result = ctx->Input("in_0");
    for (int i = 1; i < num_inputs; ++i) {
      std::string name = absl::StrCat("in_", i);
      CHECK_EQ(in_shape, ctx->InputShape(name));
      auto *layer = ctx->builder()->addElementWise(*ctx->Input(name), *result, element_wise_op);
      result = layer->getOutput(0);
    }
    ctx->SetSoleOutput(result);
  }
};

REGISTER_TRT_OP_KERNEL(Add, ElementWiseOp<nvinfer1::ElementWiseOperation::kSUM>)
    .EnableTrainPhase()
    .Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
