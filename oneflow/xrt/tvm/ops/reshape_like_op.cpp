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

namespace oneflow {
namespace xrt {
namespace of_tvm {

class ReshapeLikeOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    LOG(WARNING) << ctx->DebugStr();
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in_0"));
    node_inputs.push_back(ctx->GetExpr4InputName("like_0"));

    const Shape& x_shape = ctx->GetShape4InputName("in_0");
    const Shape& like_shape = ctx->GetShape4InputName("like_0");
    CHECK_EQ(x_shape.elem_cnt(), like_shape.elem_cnt());

    auto op = tvm::relay::Op::Get("reshape_like");
    auto expr = tvm::relay::Call(op, node_inputs, tvm::Attrs(), {});
    ctx->SetExpr4OutputName("out_0", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(ReshapeLike, ReshapeLikeOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
