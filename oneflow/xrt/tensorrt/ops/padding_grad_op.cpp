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

class PaddingGradOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext* ctx) override {
    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const auto& shape = ctx->InputShape("dy_0");
    CHECK_EQ(shape.NumAxes(), padding_before.size());
    CHECK_EQ(shape.NumAxes(), padding_after.size());
    nvinfer1::Dims start, size, stride;
    start.nbDims = padding_before.size();
    size.nbDims = start.nbDims;
    stride.nbDims = start.nbDims;
    for (int i = 0; i < start.nbDims; ++i) {
      start.d[i] = padding_before[i];
      size.d[i] = shape.At(i) - padding_before[i] - padding_after[i];
      stride.d[i] = 1;
    }
    auto* layer = ctx->builder()->addSlice(*(ctx->Input("dy_0")), start, size, stride);
    layer->setName(ctx->op_name().c_str());

    // add identity layer after slice to bypass some internal error,
    // refer to https://github.com/NVIDIA/TensorRT/issues/1821
    auto* identity_layer = ctx->builder()->addIdentity(*(layer->getOutput(0)));
    std::string name = ctx->op_name() + ".identity";
    identity_layer->setName(name.c_str());
    ctx->SetOutput("dx_0", identity_layer->getOutput(0));
  }
};

REGISTER_TRT_OP_KERNEL(PaddingGrad, PaddingGradOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
