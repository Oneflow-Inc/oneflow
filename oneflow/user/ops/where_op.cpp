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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<Shape> GetBroadcastShape(const Shape& a_shape, const Shape& b_shape) {
  Shape broadcast_shape = Shape::Ones(std::max(a_shape.NumAxes(), b_shape.NumAxes()));
  Shape a_extend_shape = CreateLeftExtendedShape(ShapeView(a_shape), broadcast_shape.NumAxes());
  Shape b_extend_shape = CreateLeftExtendedShape(ShapeView(b_shape), broadcast_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, broadcast_shape.NumAxes()) {
    CHECK_OR_RETURN(a_extend_shape.At(i) == 1 || b_extend_shape.At(i) == 1
                    || a_extend_shape.At(i) == b_extend_shape.At(i))
        << Error::RuntimeError() << "The size of tensor a (" << a_extend_shape.At(i)
        << ") must match the size of tensor b (" << b_extend_shape.At(i)
        << ") at non-singleton dimension " << i;
    broadcast_shape.Set(i, std::max(a_extend_shape.At(i), b_extend_shape.At(i)));
  }
  return broadcast_shape;
}

Maybe<std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>> CalValidSplitDims(
    const Shape& a_shape, const Shape& b_shape, const Shape& c_shape) {
  std::shared_ptr<std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>> vaild_split_dims =
      std::make_shared<std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>>();
  int32_t max_num_axes =
      std::max(a_shape.NumAxes(), std::max(b_shape.NumAxes(), c_shape.NumAxes()));
  Shape broadcast_shape = Shape::Ones(std::max(a_shape.NumAxes(), b_shape.NumAxes()));
  Shape a_extend_shape = CreateLeftExtendedShape(ShapeView(a_shape), broadcast_shape.NumAxes());
  Shape b_extend_shape = CreateLeftExtendedShape(ShapeView(b_shape), broadcast_shape.NumAxes());
  Shape c_extend_shape = CreateLeftExtendedShape(ShapeView(c_shape), broadcast_shape.NumAxes());
  int64_t a_dim_offset = max_num_axes - a_shape.NumAxes();
  int64_t b_dim_offset = max_num_axes - b_shape.NumAxes();
  int64_t c_dim_offset = max_num_axes - c_shape.NumAxes();
  FOR_RANGE(int64_t, i, 0, max_num_axes) {
    if (a_extend_shape.At(i) != 1 && a_extend_shape.At(i) == b_extend_shape.At(i)
        && a_extend_shape.At(i) == c_extend_shape.At(i)) {
      vaild_split_dims->emplace_back(
          std::make_tuple(i - a_dim_offset, i - b_dim_offset, i - c_dim_offset, i));
    }
  }
  return vaild_split_dims;
}

Maybe<std::vector<std::tuple<int64_t, int64_t, int64_t>>> CalValidSplitDims(const Shape& a_shape,
                                                                            const Shape& b_shape) {
  std::shared_ptr<std::vector<std::tuple<int64_t, int64_t, int64_t>>> vaild_split_dims =
      std::make_shared<std::vector<std::tuple<int64_t, int64_t, int64_t>>>();
  int32_t max_num_axes = std::max(a_shape.NumAxes(), b_shape.NumAxes());
  Shape broadcast_shape = Shape::Ones(std::max(a_shape.NumAxes(), b_shape.NumAxes()));
  Shape a_extend_shape = CreateLeftExtendedShape(ShapeView(a_shape), broadcast_shape.NumAxes());
  Shape b_extend_shape = CreateLeftExtendedShape(ShapeView(b_shape), broadcast_shape.NumAxes());
  int64_t a_dim_offset = max_num_axes - a_shape.NumAxes();
  int64_t b_dim_offset = max_num_axes - b_shape.NumAxes();
  FOR_RANGE(int64_t, i, 0, max_num_axes) {
    if (a_extend_shape.At(i) != 1 && a_extend_shape.At(i) == b_extend_shape.At(i)) {
      vaild_split_dims->emplace_back(std::make_tuple(i - a_dim_offset, i - b_dim_offset, i));
    }
  }
  return vaild_split_dims;
}

Maybe<void> InferWhereTensorDesc(user_op::InferContext* ctx) {
  const Shape& cond_shape = ctx->InputShape("condition", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& y_shape = ctx->InputShape("y", 0);
  if (x_shape == y_shape && y_shape == cond_shape) {
    *ctx->MutOutputShape("out", 0) = cond_shape;
  } else {
    Shape max_shape = *JUST(GetBroadcastShape(cond_shape, x_shape));
    max_shape = *JUST(GetBroadcastShape(max_shape, y_shape));
    *ctx->MutOutputShape("out", 0) = max_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereXScalarTensorDesc(user_op::InferContext* ctx) {
  const Shape& cond_shape = ctx->InputShape("condition", 0);
  const Shape& y_shape = ctx->InputShape("y", 0);
  if (cond_shape == y_shape) {
    *ctx->MutOutputShape("out", 0) = cond_shape;
  } else {
    Shape max_shape = *JUST(GetBroadcastShape(cond_shape, y_shape));
    *ctx->MutOutputShape("out", 0) = max_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereYScalarTensorDesc(user_op::InferContext* ctx) {
  const Shape& cond_shape = ctx->InputShape("condition", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  if (cond_shape == x_shape) {
    *ctx->MutOutputShape("out", 0) = cond_shape;
  } else {
    Shape max_shape = *JUST(GetBroadcastShape(cond_shape, x_shape));
    *ctx->MutOutputShape("out", 0) = max_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereXYScalarTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("out", 0) = ctx->InputShape("condition", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& cond_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0).shape();
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  const auto& vaild_split_dims = JUST(CalValidSplitDims(cond_shape, x_shape, y_shape));
  for (const auto& vaild_split_dim : *vaild_split_dims) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("condition", 0), std::get<0>(vaild_split_dim))
        .Split(user_op::OpArg("x", 0), std::get<1>(vaild_split_dim))
        .Split(user_op::OpArg("y", 0), std::get<2>(vaild_split_dim))
        .Split(user_op::OpArg("out", 0), std::get<3>(vaild_split_dim))
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("condition", 0))
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereXScalarSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& cond_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  const auto& vaild_split_dims = JUST(CalValidSplitDims(cond_shape, y_shape));
  for (const auto& vaild_split_dim : *vaild_split_dims) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("condition", 0), std::get<0>(vaild_split_dim))
        .Split(user_op::OpArg("y", 0), std::get<1>(vaild_split_dim))
        .Split(user_op::OpArg("out", 0), std::get<2>(vaild_split_dim))
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("condition", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereYScalarSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& cond_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0).shape();
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const auto& vaild_split_dims = JUST(CalValidSplitDims(cond_shape, x_shape));
  for (const auto& vaild_split_dim : *vaild_split_dims) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("condition", 0), std::get<0>(vaild_split_dim))
        .Split(user_op::OpArg("x", 0), std::get<1>(vaild_split_dim))
        .Split(user_op::OpArg("out", 0), std::get<2>(vaild_split_dim))
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("condition", 0))
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereXYScalarSbpSignatures(user_op::SbpContext* ctx) {
  const Shape& cond_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0).shape();
  FOR_RANGE(int64_t, i, 0, cond_shape.NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("condition", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereInputArgModify(const GetInputArgModifier& GetInputArgModifierFn,
                                   const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* cond_arg_modifier = GetInputArgModifierFn("condition", 0);
  cond_arg_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> WhereOp::GetSbp(user_op::SbpContext* ctx) {
  return GetWhereSbpSignatures(ctx);
}
/*static*/ Maybe<void> WhereOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferWhereTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereOp::InferDataType(user_op::InferContext* ctx) {
  DataType cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  DataType x_dtype = ctx->InputDType("x", 0);
  CHECK_EQ_OR_RETURN(x_dtype, ctx->InputDType("y", 0));
  *ctx->MutOutputDType("out", 0) = x_dtype;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WhereOp::ModifyInputArg(const GetInputArgModifier& f,
                                               const user_op::UserOpConfWrapper& conf) {
  return GetWhereInputArgModify(f, conf);
}

/*static*/ Maybe<void> WhereScalarXOp::GetSbp(user_op::SbpContext* ctx) {
  return GetWhereXScalarSbpSignatures(ctx);
}
/*static*/ Maybe<void> WhereScalarXOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferWhereXScalarTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereScalarXOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereScalarXOp::InferDataType(user_op::InferContext* ctx) {
  DataType cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  DataType y_dtype = ctx->InputDType("y", 0);
  if (ctx->Attr<bool>("has_int_operand")) {
    CHECK_EQ_OR_RETURN(y_dtype, GetDataType<int64_t>::value)
        << "expected scalar type " << GetDataType<int64_t>::value << "but found " << y_dtype;
  } else if (ctx->Attr<bool>("has_float_operand")) {
    CHECK_EQ_OR_RETURN(y_dtype, GetDataType<double>::value)
        << "expected scalar type " << GetDataType<double>::value << "but found " << y_dtype;
  } else if (ctx->Attr<bool>("has_bool_operand")) {
    CHECK_EQ_OR_RETURN(y_dtype, GetDataType<bool>::value)
        << "expected scalar type " << GetDataType<bool>::value << "but found " << y_dtype;
  }
  *ctx->MutOutputDType("out", 0) = y_dtype;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WhereScalarXOp::ModifyInputArg(const GetInputArgModifier& f,
                                                      const user_op::UserOpConfWrapper& conf) {
  return GetWhereInputArgModify(f, conf);
}

/*static*/ Maybe<void> WhereScalarYOp::GetSbp(user_op::SbpContext* ctx) {
  return GetWhereYScalarSbpSignatures(ctx);
}
/*static*/ Maybe<void> WhereScalarYOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferWhereYScalarTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereScalarYOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereScalarYOp::InferDataType(user_op::InferContext* ctx) {
  DataType cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  DataType x_dtype = ctx->InputDType("x", 0);
  if (ctx->Attr<bool>("has_int_operand")) {
    CHECK_EQ_OR_RETURN(x_dtype, GetDataType<int64_t>::value)
        << "expected scalar type " << GetDataType<int64_t>::value << "but found " << x_dtype;
  } else if (ctx->Attr<bool>("has_float_operand")) {
    CHECK_EQ_OR_RETURN(x_dtype, GetDataType<double>::value)
        << "expected scalar type " << GetDataType<double>::value << "but found " << x_dtype;
  } else if (ctx->Attr<bool>("has_bool_operand")) {
    CHECK_EQ_OR_RETURN(x_dtype, GetDataType<bool>::value)
        << "expected scalar type " << GetDataType<bool>::value << "but found " << x_dtype;
  }
  *ctx->MutOutputDType("out", 0) = x_dtype;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WhereScalarYOp::ModifyInputArg(const GetInputArgModifier& f,
                                                      const user_op::UserOpConfWrapper& conf) {
  return GetWhereInputArgModify(f, conf);
}

/*static*/ Maybe<void> WhereScalarXyOp::GetSbp(user_op::SbpContext* ctx) {
  return GetWhereXYScalarSbpSignatures(ctx);
}
/*static*/ Maybe<void> WhereScalarXyOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferWhereXYScalarTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereScalarXyOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> WhereScalarXyOp::InferDataType(user_op::InferContext* ctx) {
  DataType cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  if (ctx->Attr<bool>("has_x_bool_operand") && ctx->Attr<bool>("has_y_bool_operand")) {
    *ctx->MutOutputDType("out", 0) = GetDataType<bool>::value;
  } else if (ctx->Attr<bool>("has_x_int_operand") && ctx->Attr<bool>("has_y_int_operand")) {
    *ctx->MutOutputDType("out", 0) = GetDataType<int64_t>::value;
  } else if (ctx->Attr<bool>("has_x_float_operand") && ctx->Attr<bool>("has_y_float_operand")) {
    *ctx->MutOutputDType("out", 0) = GetDataType<double>::value;
  } else {
    UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> WhereScalarXyOp::ModifyInputArg(const GetInputArgModifier& f,
                                                       const user_op::UserOpConfWrapper& conf) {
  return GetWhereInputArgModify(f, conf);
}

REGISTER_USER_OP_GRAD("where").SetBackwardOpConfGenFn(
    [](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto zero_op_name = ctx->FwOp().op_name() + "_zero_grad";
      ctx->DefineOp(zero_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("zero_like")
            .InputBind("like", ctx->FwOp().input("x", 0))
            .Output("out")
            .Build();
      });

      const auto x_grad_op_name = ctx->FwOp().op_name() + "_x_grad";
      ctx->DefineOp(x_grad_op_name, [&ctx, &zero_op_name](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("where")
            .InputBind("condition", ctx->FwOp().input("condition", 0))
            .InputBind("x", ctx->FwOp().output_grad("out", 0))
            .InputBind("y", ctx->GetOp(zero_op_name).output("out", 0))
            .Output("out")
            .Build();
      });

      const auto y_grad_op_name = ctx->FwOp().op_name() + "_y_grad";
      ctx->DefineOp(y_grad_op_name, [&ctx, &zero_op_name](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("where")
            .InputBind("condition", ctx->FwOp().input("condition", 0))
            .InputBind("x", ctx->GetOp(zero_op_name).output("out", 0))
            .InputBind("y", ctx->FwOp().output_grad("out", 0))
            .Output("out")
            .Build();
      });

      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                                [&ctx, &x_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(x_grad_op_name).output("out", 0);
                                });
      ctx->FwOp().InputGradBind(user_op::OpArg("y", 0),
                                [&ctx, &y_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(y_grad_op_name).output("out", 0);
                                });
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
