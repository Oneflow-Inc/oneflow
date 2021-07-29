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
#include "oneflow/xrt/tvm/ops/op_kernel.h"
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

tvm::relay::Expr Transpose2D(tvm::relay::Expr input, const bool is_transpose) {
  if (is_transpose) {
    auto transpose_op = tvm::relay::Op::Get("transpose");
    tvm::Array<tvm::Integer> tvm_axes{1, 0};
    auto transpose_attrs = tvm::runtime::make_object<tvm::relay::TransposeAttrs>();
    transpose_attrs->axes = std::move(tvm_axes);
    auto expr = tvm::relay::Call(transpose_op, tvm::Array<tvm::relay::Expr>({input}),
                                 tvm::Attrs(transpose_attrs), {});
    return expr;
  }

  return input;
}

class MatMulOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    Shape a_shape = ctx->GetShape4InputName("a_0");
    Shape b_shape = ctx->GetShape4InputName("b_0");
    CHECK_GE(a_shape.NumAxes(), 2);
    CHECK_EQ(a_shape.NumAxes(), b_shape.NumAxes());

    tvm::Array<tvm::relay::Expr> node_inputs;
    bool transpose_a = ctx->Attr<bool>("transpose_a");
    bool transpose_b = ctx->Attr<bool>("transpose_b") ? false : true;

    auto a = ctx->GetExpr4InputName("a_0");
    auto b = ctx->GetExpr4InputName("b_0");

    node_inputs.push_back(Transpose2D(a, transpose_a));
    node_inputs.push_back(Transpose2D(b, transpose_b));

    auto matmul_op = tvm::relay::Op::Get("nn.dense");
    auto expr =
        tvm::relay::Call(matmul_op, node_inputs,
                         tvm::Attrs(tvm::runtime::make_object<tvm::relay::DenseAttrs>()), {});
    ctx->SetExpr4OutputName("out_0", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(MatMul, MatMulOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
