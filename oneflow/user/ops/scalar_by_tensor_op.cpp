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

Maybe<void> TensorDescInferFn(user_op::InferContext* ctx) {
  const user_op::TensorDesc* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* scalar = ctx->TensorDesc4ArgNameAndIndex("scalar", 0);
  CHECK_EQ_OR_RETURN(x->data_type(), scalar->data_type());
  CHECK_EQ_OR_RETURN(scalar->shape().elem_cnt(), 1);
  user_op::TensorDesc* y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  *y = *x;
  return Maybe<void>::Ok();
}

Maybe<void> GetBasicSbpSignature(user_op::SbpContext* ctx) {
  const auto& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Broadcast(user_op::OpArg("scalar", 0))
        .Build();
  }
  return Maybe<void>::Ok();
}

using GetSbpFn = std::function<Maybe<void>(user_op::SbpContext*)>;
GetSbpFn MakeGetSbpFn(GetSbpFn extra) {
  return [extra](user_op::SbpContext* ctx) -> Maybe<void> {
    JUST(extra(ctx));
    GetBasicSbpSignature(ctx);
    return Maybe<void>::Ok();
  };
}

}  // namespace

REGISTER_USER_OP("scalar_add_by_tensor")
    .Input("x")
    .Input("scalar")
    .Output("y")
    .SetTensorDescInferFn(TensorDescInferFn)
    .SetGetSbpFn(MakeGetSbpFn([](user_op::SbpContext* ctx) {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("x", 0))
          .PartialSum(user_op::OpArg("scalar", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
      return Maybe<void>::Ok();
    }));

REGISTER_USER_OP_GRAD("scalar_add_by_tensor")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("y", 0), "x", 0);
      }
      if (op.NeedGenGradTensor4OpInput("scalar", 0)) {
        std::vector<int32_t> axes_vec(op.TensorDesc4ArgNameAndIndex("y", 0).shape().NumAxes());
        std::iota(axes_vec.begin(), axes_vec.end(), 0);
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "scalar_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("reduce_sum")
                .Input("input_tensor", op.GetGradTensorWithOpOutput("y", 0))
                .Output("output_tensor")
                .Attr("axis", axes_vec)
                .Attr("keepdims", false)
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("output_tensor", 0), "scalar", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP("scalar_sub_by_tensor")
    .Input("x")
    .Input("scalar")
    .Output("y")
    .SetTensorDescInferFn(TensorDescInferFn)
    .SetGetSbpFn(MakeGetSbpFn([](user_op::SbpContext* ctx) {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("x", 0))
          .PartialSum(user_op::OpArg("scalar", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
      return Maybe<void>::Ok();
    }));

REGISTER_USER_OP_GRAD("scalar_sub_by_tensor")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        op.BindGradTensorWithOpInput(op.GetGradTensorWithOpOutput("y", 0), "x", 0);
      }
      if (op.NeedGenGradTensor4OpInput("scalar", 0)) {
        std::vector<int32_t> axes_vec(op.TensorDesc4ArgNameAndIndex("y", 0).shape().NumAxes());
        std::iota(axes_vec.begin(), axes_vec.end(), 0);
        user_op::UserOpConfWrapperBuilder builder0(op.op_name() + "scalar_grad_reduce_sum");
        user_op::UserOpConfWrapper scalar_grad_reduce_sum_op =
            builder0.Op("reduce_sum")
                .Input("input_tensor", op.GetGradTensorWithOpOutput("y", 0))
                .Output("output_tensor")
                .Attr("axis", axes_vec)
                .Attr("keepdims", false)
                .Build();
        user_op::UserOpConfWrapperBuilder builder1(op.op_name() + "scalar_grad_scalar_mul");
        user_op::UserOpConfWrapper scalar_grad_scalar_mul_op =
            builder1.Op("scalar_mul")
                .Input("in", scalar_grad_reduce_sum_op.output("output_tensor", 0))
                .Output("out")
                .Attr("has_float_operand", true)
                .Attr("has_int_operand", false)
                .Attr("float_operand", static_cast<double>(-1))
                .Attr("int_operand", static_cast<int64_t>(-1))
                .Build();
        op.BindGradTensorWithOpInput(scalar_grad_scalar_mul_op.output("out", 0), "scalar", 0);
        AddOp(scalar_grad_reduce_sum_op);
        AddOp(scalar_grad_scalar_mul_op);
      }
    });

REGISTER_USER_OP("scalar_mul_by_tensor")
    .Input("x")
    .Input("scalar")
    .Output("y")
    .SetTensorDescInferFn(TensorDescInferFn)
    .SetGetSbpFn(MakeGetSbpFn([](user_op::SbpContext* ctx) {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("scalar", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .PartialSum(user_op::OpArg("scalar", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
      return Maybe<void>::Ok();
    }));

REGISTER_USER_OP_GRAD("scalar_mul_by_tensor")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("scalar_mul_by_tensor")
                                                 .Input("x", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Input("scalar", op.input("scalar", 0))
                                                 .Output("y")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "x", 0);
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("scalar", 0)) {
        int64_t num_axes = op.TensorDesc4ArgNameAndIndex("y", 0).shape().NumAxes();
        user_op::UserOpConfWrapperBuilder builder0(op.op_name() + "scalar_grad_multiply");
        user_op::UserOpConfWrapper scalar_grad_multiply_op =
            builder0.Op("multiply")
                .Input("x", op.GetGradTensorWithOpOutput("y", 0))
                .Input("y", op.input("x", 0))
                .Output("out")
                .Build();
        std::vector<int32_t> axes_vec(num_axes);
        std::iota(axes_vec.begin(), axes_vec.end(), 0);
        user_op::UserOpConfWrapperBuilder builder1(op.op_name() + "scalar_grad_reduce_sum");
        user_op::UserOpConfWrapper scalar_grad_reduce_sum_op =
            builder1.Op("reduce_sum")
                .Input("input_tensor", scalar_grad_multiply_op.output("out", 0))
                .Output("output_tensor")
                .Attr("axis", axes_vec)
                .Attr("keepdims", false)
                .Build();
        op.BindGradTensorWithOpInput(scalar_grad_reduce_sum_op.output("output_tensor", 0), "scalar",
                                     0);
        AddOp(scalar_grad_multiply_op);
        AddOp(scalar_grad_reduce_sum_op);
      }
    });

REGISTER_USER_OP("scalar_div_by_tensor")
    .Input("x")
    .Input("scalar")
    .Output("y")
    .SetTensorDescInferFn(TensorDescInferFn)
    .SetGetSbpFn(MakeGetSbpFn([](user_op::SbpContext* ctx) {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("x", 0))
          .Broadcast(user_op::OpArg("scalar", 0))
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
      return Maybe<void>::Ok();
    }));

REGISTER_USER_OP_GRAD("scalar_div_by_tensor")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("scalar_div_by_tensor")
                                                 .Input("x", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Input("scalar", op.input("scalar", 0))
                                                 .Output("y")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("y", 0), "x", 0);
        AddOp(grad_op);
      }
      if (op.NeedGenGradTensor4OpInput("scalar", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "scalar_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("broadcast_div_grad")
                                                 .Input("dz", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Input("z", op.output("y", 0))
                                                 .Input("y", op.input("scalar", 0))
                                                 .Output("dy")
                                                 .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dy", 0), "scalar", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
