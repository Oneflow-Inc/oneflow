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

REGISTER_USER_OP("narrow")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("dim")
    .Attr<int64_t>("start")
    .Attr<int64_t>("length")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
      CHECK_GT_OR_RETURN(in.shape().NumAxes(), 0);
      const int64_t& dim = ctx->Attr<int64_t>("dim");
      const int64_t& start = ctx->Attr<int64_t>("start");
      const int64_t& length = ctx->Attr<int64_t>("length");
      CHECK_GE_OR_RETURN(dim, 0);
      CHECK_GE_OR_RETURN(start, 0);
      CHECK_GE_OR_RETURN(length, 0);
      CHECK_GE_OR_RETURN(in.shape().At(dim), start + length);
      user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);

      DimVector dim_vec;
      dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin(),
                     in.shape().dim_vec().cbegin() + dim);
      dim_vec.insert(dim_vec.end(), length);
      dim_vec.insert(dim_vec.end(), in.shape().dim_vec().cbegin() + dim + 1,
                     in.shape().dim_vec().end());
      *out->mut_shape() = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      const int64_t& dim = ctx->Attr<int64_t>("dim");
      const int64_t& start = ctx->Attr<int64_t>("start");
      const int64_t& length = ctx->Attr<int64_t>("length");
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
        if (i != dim) {
          ctx->NewBuilder()
              .Split(user_op::OpArg("in", 0), i)
              .Split(user_op::OpArg("out", 0), i)
              .Build();
        } else {
          if (length == in_tensor.shape().At(i)) {
            ctx->NewBuilder()
                .Split(user_op::OpArg("in", 0), i)
                .Split(user_op::OpArg("out", 0), i)
                .Build();
          }
        }
      }
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .PartialSum(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->InputTensorDesc("in", 0);
      user_op::TensorDesc* out = ctx->OutputTensorDesc("out", 0);
      *out->mut_data_type() = in.data_type();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("narrow_grad")
    .Input("dy")
    .Input("like")
    .Output("dx")
    .Attr<int64_t>("dim")
    .Attr<int64_t>("start")
    .Attr<int64_t>("length")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& like_shape = ctx->InputShape("like", 0);
      const Shape& dy_shape = ctx->InputShape("dy", 0);
      const int64_t ndim = dy_shape.NumAxes();
      CHECK_EQ_OR_RETURN(like_shape.NumAxes(), ndim);

      *ctx->OutputShape("dx", 0) = like_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Shape& like_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
      const int64_t ndim = like_shape.NumAxes();
      const int64_t& dim = ctx->Attr<int64_t>("dim");
      const int64_t& length = ctx->Attr<int64_t>("length");
      FOR_RANGE(int64_t, i, 0, ndim) {
        if (i != dim) {
          ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
        } else {
          if (length == like_shape.At(i)) {
            ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
          }
        }
      }
      ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("dy", 0))
          .Broadcast(user_op::OpArg("like", 0))
          .PartialSum(user_op::OpArg("dx", 0))
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("dy", 0))
          .PartialSum(user_op::OpArg("like", 0))
          .Broadcast(user_op::OpArg("dx", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
      user_op::InputArgModifier* dy_modifier = GetInputArgModifierFn("dy", 0);
      CHECK_NOTNULL_OR_RETURN(dy_modifier);
      dy_modifier->set_requires_grad(false);
      user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
      CHECK_NOTNULL_OR_RETURN(like_modifier);
      like_modifier->set_requires_grad(false);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("narrow").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder in_grad_builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper in_grad_op = in_grad_builder.Op("narrow_grad")
                                                .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
                                                .Input("like", op.input("in", 0))
                                                .Attr("dim", op.attr<int64_t>("dim"))
                                                .Attr("start", op.attr<int64_t>("start"))
                                                .Attr("length", op.attr<int64_t>("length"))
                                                .Output("dx")
                                                .Build();
    op.BindGradTensorWithOpInput(in_grad_op.output("out", 0), "in", 0);
    AddOp(in_grad_op);
  }
  return Maybe<void>::Ok();
});

}  // namespace oneflow
