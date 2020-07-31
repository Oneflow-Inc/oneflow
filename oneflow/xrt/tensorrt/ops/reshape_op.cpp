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

class ReshapeOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    Shape shape = ctx->SoleOutputShape();
    CHECK_EQ(shape.Count(0), in_shape.Count(0));

    nvinfer1::ITensor *input = ctx->SoleInput();
    ctx->SetSoleOutput(helpers::Reshape(ctx, input, shape));
  }
};

REGISTER_TRT_OP_KERNEL(Reshape, ReshapeOp).EnableTrainPhase().Finalize();

class ReshapeLikeOp : public TrtOpKernel {
 public:
  void Compile(TrtOpContext *ctx) override {
    Shape x_shape = ctx->InputShape("in_0");
    Shape like_shape = ctx->InputShape("like_0");
    CHECK_EQ(x_shape.Count(0), like_shape.Count(0));

    nvinfer1::ITensor *input = ctx->Input("in_0");
    ctx->SetSoleOutput(helpers::Reshape(ctx, input, like_shape));
  }
};

REGISTER_TRT_OP_KERNEL(ReshapeLike, ReshapeLikeOp).EnableTrainPhase().Finalize();

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
