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

Maybe<void> InferWhereTensorDesc(user_op::InferContext* ctx) {
  const Shape& cond_shape = ctx->InputShape("condition", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& y_shape = ctx->InputShape("y", 0);
  if (x_shape == y_shape && y_shape == cond_shape) {
    *ctx->OutputShape("out", 0) = cond_shape;
  } else {
    Shape max_shape =
        Shape::Ones(std::max(x_shape.NumAxes(), std::max(y_shape.NumAxes(), cond_shape.NumAxes())));
    const Shape& x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), max_shape.NumAxes());
    const Shape& y_extend_shape = CreateLeftExtendedShape(ShapeView(y_shape), max_shape.NumAxes());
    const Shape& cond_extend_shape =
        CreateLeftExtendedShape(ShapeView(cond_shape), max_shape.NumAxes());
    FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
      max_shape.Set(i, std::max(x_extend_shape.At(i),
                                std::max(y_extend_shape.At(i), cond_extend_shape.At(i))));
    }
    *ctx->OutputShape("out", 0) = max_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereXScalarTensorDesc(user_op::InferContext* ctx) {
  const Shape& cond_shape = ctx->InputShape("condition", 0);
  const Shape& y_shape = ctx->InputShape("y", 0);
  if (cond_shape == y_shape) {
    *ctx->OutputShape("out", 0) = cond_shape;
  } else {
    Shape max_shape = Shape::Ones(std::max(y_shape.NumAxes(), cond_shape.NumAxes()));
    const Shape& y_extend_shape = CreateLeftExtendedShape(ShapeView(y_shape), max_shape.NumAxes());
    const Shape& cond_extend_shape =
        CreateLeftExtendedShape(ShapeView(cond_shape), max_shape.NumAxes());
    FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
      max_shape.Set(i, std::max(y_extend_shape.At(i), cond_extend_shape.At(i)));
    }
    *ctx->OutputShape("out", 0) = max_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereYScalarTensorDesc(user_op::InferContext* ctx) {
  const Shape& cond_shape = ctx->InputShape("condition", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  if (cond_shape == x_shape) {
    *ctx->OutputShape("out", 0) = cond_shape;
  } else {
    Shape max_shape = Shape::Ones(std::max(x_shape.NumAxes(), cond_shape.NumAxes()));
    const Shape& x_extend_shape = CreateLeftExtendedShape(ShapeView(x_shape), max_shape.NumAxes());
    const Shape& cond_extend_shape =
        CreateLeftExtendedShape(ShapeView(cond_shape), max_shape.NumAxes());
    FOR_RANGE(int64_t, i, 0, max_shape.NumAxes()) {
      max_shape.Set(i, std::max(x_extend_shape.At(i), cond_extend_shape.At(i)));
    }
    *ctx->OutputShape("out", 0) = max_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferWhereXYScalarTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("condition", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetWhereSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& condition_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("condition", 0);
  FOR_RANGE(int64_t, i, 0, condition_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("condition", 0), i)
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("out", 0), i)
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
  if (cond_shape.NumAxes() < y_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, y_shape.NumAxes() - cond_shape.NumAxes()) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("condition", 0))
          .Split(user_op::OpArg("y", 0), i)
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, cond_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("condition", 0), cond_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("y", 0), y_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), y_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else if (cond_shape.NumAxes() > y_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, cond_shape.NumAxes() - y_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("condition", 0), i)
          .Broadcast(user_op::OpArg("y", 0))
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, y_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("condition", 0), cond_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("y", 0), y_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), cond_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else {
    FOR_RANGE(int64_t, i, 0, cond_shape.NumAxes()) {
      if (cond_shape.At(i) == 1 && y_shape.At(i) == 1) { continue; }
      if (cond_shape.At(i) == y_shape.At(i)) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      } else if (cond_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Broadcast(user_op::OpArg("condition", 0))
            .Split(user_op::OpArg("y", 0), i)
            .Split(ctx->outputs(), i)
            .Build();
      } else if (y_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("condition", 0), i)
            .Broadcast(user_op::OpArg("y", 0))
            .Split(ctx->outputs(), i)
            .Build();
      } else {
        UNIMPLEMENTED();
      }
    }
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
  if (cond_shape.NumAxes() < x_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes() - cond_shape.NumAxes()) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("condition", 0))
          .Split(user_op::OpArg("x", 0), i)
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, cond_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("condition", 0), cond_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), x_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else if (cond_shape.NumAxes() > x_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, cond_shape.NumAxes() - x_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("condition", 0), i)
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("out", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("condition", 0), cond_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), cond_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else {
    FOR_RANGE(int64_t, i, 0, cond_shape.NumAxes()) {
      if (cond_shape.At(i) == 1 && x_shape.At(i) == 1) { continue; }
      if (cond_shape.At(i) == x_shape.At(i)) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      } else if (cond_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Broadcast(user_op::OpArg("condition", 0))
            .Split(user_op::OpArg("x", 0), i)
            .Split(ctx->outputs(), i)
            .Build();
      } else if (x_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("condition", 0), i)
            .Broadcast(user_op::OpArg("x", 0))
            .Split(ctx->outputs(), i)
            .Build();
      } else {
        UNIMPLEMENTED();
      }
    }
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
  const DataType& cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  const DataType& x_dtype = ctx->InputDType("x", 0);
  CHECK_EQ_OR_RETURN(x_dtype, ctx->InputDType("y", 0));
  *ctx->OutputDType("out", 0) = x_dtype;
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
  const DataType& cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  const DataType& y_dtype = ctx->InputDType("y", 0);
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
  *ctx->OutputDType("out", 0) = y_dtype;
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
  const DataType& cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  const DataType& x_dtype = ctx->InputDType("x", 0);
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
  *ctx->OutputDType("out", 0) = x_dtype;
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
  const DataType& cond_dtype = ctx->InputDType("condition", 0);
  CHECK_OR_RETURN(IsBoolDataType(cond_dtype) || IsIntegralDataType(cond_dtype));
  if (ctx->Attr<bool>("has_x_bool_operand") && ctx->Attr<bool>("has_y_bool_operand")) {
    *ctx->OutputDType("out", 0) = GetDataType<bool>::value;
  } else if (ctx->Attr<bool>("has_x_int_operand") && ctx->Attr<bool>("has_y_int_operand")) {
    *ctx->OutputDType("out", 0) = GetDataType<int64_t>::value;
  } else if (ctx->Attr<bool>("has_x_float_operand") && ctx->Attr<bool>("has_y_float_operand")) {
    *ctx->OutputDType("out", 0) = GetDataType<double>::value;
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
