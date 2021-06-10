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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc* first_in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const int64_t axis = ctx->Attr<int64_t>("axis");
  
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("adaptive_avg_pool2d")
    .Input("in")
    .Attr<std::vector<int>>("output_size")
    .Output("out")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("adaptive_avg_pool2d_grad")
    .Input("x")
    .Input("dy")
    .Attr<std::vector<int>>("output_size")
    .Output("dx")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP_GRAD("adaptive_avg_pool2d").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto adaptive_avg_pool2d_grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(adaptive_avg_pool2d_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("adaptive_avg_pool2d_grad")
        .InputBind("x", ctx->FwOp().input("in", 0))
        .InputBind("dy", ctx->FwOp().output_grad("out", 0))
        .Attr("scale", ctx->FwOp().attr<double>("scale"))
        .Attr("alpha", ctx->FwOp().attr<double>("alpha"))
        .Output("dx")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("in", 0),
                            [&ctx, &adaptive_avg_pool2d_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(adaptive_avg_pool2d_grad_op_name).output("dx", 0);
                            });
});

}  // namespace

}  // namespace oneflow
