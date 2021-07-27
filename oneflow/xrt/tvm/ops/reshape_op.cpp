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
#include <tvm/relay/attrs/transform.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class ReshapeOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    VLOG(3) << ctx->DebugStr();
    tvm::Array<tvm::relay::Expr> node_inputs;
    node_inputs.push_back(ctx->GetExpr4InputName("in_0"));
    const Shape& in_shape = ctx->GetShape4InputName("in_0");

    auto reshape_attrs = tvm::runtime::make_object<tvm::relay::ReshapeAttrs>();
    {
      const Shape& conf_shape = ctx->GetShape4OutputName("out_0");
      CHECK_EQ(in_shape.elem_cnt(), conf_shape.elem_cnt());
      tvm::Array<tvm::Integer> tvm_conf_shape;
      for (int64_t dim : conf_shape.dim_vec()) { tvm_conf_shape.push_back(static_cast<int>(dim)); }
      reshape_attrs->newshape = std::move(tvm_conf_shape);
    }

    auto op = tvm::relay::Op::Get("reshape");
    auto expr = tvm::relay::Call(op, node_inputs, tvm::Attrs(reshape_attrs), {});
    ctx->SetExpr4OutputName("out_0", std::move(expr));
  }
};

REGISTER_TVM_OP_KERNEL(Reshape, ReshapeOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
