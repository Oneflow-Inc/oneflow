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
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

class ReshapeOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext* ctx) override {
    Shape in_shape = ctx->SoleInputShape();
    Shape shape = ctx->SoleOutputShape();
    CHECK_EQ(shape.Count(0), in_shape.Count(0));

    ctx->SetSoleOutput(Reshape(ctx->SoleInput(), shape));
  }
};

REGISTER_XLA_OP_KERNEL(Reshape, ReshapeOp).Finalize();

class ReshapeLikeOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext* ctx) override {
    Shape x_shape = ctx->InputShape("in_0");
    Shape like_shape = ctx->InputShape("like_0");
    CHECK_EQ(x_shape.Count(0), like_shape.Count(0));

    ctx->SetOutput("out_0", Reshape(ctx->Input("in_0"), like_shape));
  }
};

REGISTER_XLA_OP_KERNEL(ReshapeLike, ReshapeLikeOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
