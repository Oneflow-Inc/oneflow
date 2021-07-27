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

namespace oneflow {
namespace xrt {
namespace of_tvm {

class BatchNormOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    VLOG(3) << ctx->DebugStr();

    tvm::Array<tvm::relay::Expr> inputs;
    inputs.push_back(ctx->GetExpr4InputName("x_0"));
    inputs.push_back(ctx->GetExpr4InputName("gamma_0"));
    inputs.push_back(ctx->GetExpr4InputName("beta_0"));
    inputs.push_back(ctx->GetExpr4InputName("moving_mean_0"));
    inputs.push_back(ctx->GetExpr4InputName("moving_variance_0"));
    // TODO: handle training
    auto bn_attrs = tvm::runtime::make_object<tvm::relay::BatchNormAttrs>();
    bn_attrs->axis = ctx->Attr<int32_t>("axis");
    bn_attrs->epsilon = ctx->Attr<float>("epsilon");
    bn_attrs->center = true;
    bn_attrs->scale = true;

    const auto& bn_op = tvm::relay::Op::Get("nn.batch_norm");
    auto bn = tvm::relay::Call(bn_op, inputs, tvm::Attrs(bn_attrs), {});

    // // There is no TOpPattern attr registered for nn.batch_norm, which leads to the attr missing
    // // error when we call relay.build().
    // // But nn.batch_norm always get unpacked by SimplifyInference Pass in tvm,
    // // and SimplifyInference takes effect only when we solely need the 1st output of bn.
    // // Thus, we should return the 1st output of nn.batch_norm instead of itself here.
    auto n = tvm::relay::TupleGetItem(std::move(bn), 0);
    ctx->SetExpr4OutputName("y_0", std::move(n));
  }
};

REGISTER_TVM_OP_KERNEL(Normalization, BatchNormOp).Finalize();

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow
