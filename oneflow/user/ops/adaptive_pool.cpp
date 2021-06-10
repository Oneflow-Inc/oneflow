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
  std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
  DimVector out_dim_vec = first_in_desc->shape().dim_vec();
  int h = out_dim_vec[2];
  int w = out_dim_vec[3];
  if(output_size.size() >= 1){
    h = output_size[0]; //h
    w = output_size[0];
    if(output_size.size() == 2){
      h = output_size[1]; //w
    }
  }
  out_dim_vec[2] = h;
  out_dim_vec[3] = w;
  user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  *out_desc->mut_shape() = Shape(out_dim_vec);
  *out_desc->mut_is_dynamic() = *ctx->IsDynamic4ArgNameAndIndex("x", 0);
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  // only for nchw
  FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("adaptive_avg_pool2d")
    .Input("in")
    .Attr<std::vector<int64_t>>("output_size")
    .Output("out")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP("adaptive_avg_pool2d_grad")
    .Input("x")
    .Input("dy")
    .Attr<std::vector<int64_t>>("output_size")
    .Output("dx")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(FwGetSbpFn);

REGISTER_USER_OP_GRAD("adaptive_avg_pool2d").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto adaptive_avg_pool2d_grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(adaptive_avg_pool2d_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("adaptive_avg_pool2d_grad")
        .InputBind("x", ctx->FwOp().input("in", 0))
        .InputBind("dy", ctx->FwOp().output_grad("out", 0))
        .Attr("output_size", ctx->FwOp().attr<double>("scale"))
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
