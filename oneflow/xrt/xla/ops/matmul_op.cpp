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

namespace oneflow {
namespace xrt {
namespace mola {

class MatMulOp : public XlaOpKernel {
 public:
  void Compile(XlaOpContext *ctx) override {
    Shape a_shape = ctx->InputShape("a_0");
    Shape b_shape = ctx->InputShape("b_0");
    CHECK_GE(a_shape.NumAxes(), 2);
    CHECK_EQ(a_shape.NumAxes(), b_shape.NumAxes());

    if (a_shape.NumAxes() > 2) {
      auto batch_matmul_kernel = BuildOpKernel(ctx->device(), "BatchMatMul");
      batch_matmul_kernel->Compile(ctx);
      return;
    }

    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b");

    xla::XlaOp a = ctx->Input("a_0");
    xla::XlaOp b = ctx->Input("b_0");

    auto lhs = transpose_a ? xla::Transpose(a, {1, 0}) : a;
    auto rhs = transpose_b ? xla::Transpose(b, {1, 0}) : b;
    xla::XlaOp out = xla::Dot(lhs, rhs);
    if (ctx->HasInput("_add_to_output_0")) {
      out = xla::Add(out, ctx->Input("_add_to_output_0"));
    } 
    ctx->SetOutput("out_0", out);
  }
};

REGISTER_XLA_OP_KERNEL(MatMul, MatMulOp).Finalize();

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
