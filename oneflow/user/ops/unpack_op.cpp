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

namespace oneflow {

namespace {

REGISTER_USER_OP("unpack")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("unpack_num")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      const Shape& in_shape = in_desc->shape();
      CHECK_GT(in_shape.NumAxes(), 0);
      const auto unpack_num = ctx->Attr<int32_t>("unpack_num");
      CHECK_EQ(in_shape.At(0) % unpack_num, 0);
      user_op::TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_desc = *in_desc;
      out_desc->mut_shape()->Set(0, in_shape.At(0) / unpack_num);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetInferOutputBlobTimeShapeFn(
        [](user_op::InferOutputBlobTimeShapeFnContext* ctx) -> Maybe<void> {
          const int32_t unpack_num = ctx->user_op_conf().attr<int32_t>("unpack_num");
          DimVector time_shape_dim_vec = ctx->TimeShape4InputArgNameAndIndex("in", 0).dim_vec();
          time_shape_dim_vec.push_back(unpack_num);
          *ctx->mut_output_blob_time_shape() = Shape(time_shape_dim_vec);
          return Maybe<void>::Ok();
        });

REGISTER_USER_OP_GRAD("unpack").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("pack")
        .InputBind("in", ctx->FwOp().output_grad("out", 0))
        .Output("out")
        .Attr<int32_t>("pack_num", ctx->FwOp().attr<int32_t>("unpack_num"))
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), [&ctx, &grad_op_name]() -> const std::string& {
    return ctx->GetOp(grad_op_name).output("out", 0);
  });
});

}  // namespace

}  // namespace oneflow
