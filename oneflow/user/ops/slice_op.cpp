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
#include "oneflow/user/kernels/slice_util.h"

namespace oneflow {

namespace {

bool IsFullSlice(int64_t start, int64_t stop, int64_t step, int64_t size) {
  if (step != 1) { return false; }
  if (start != 0) { return false; }
  if (stop != std::numeric_limits<int64_t>::max()) { return false; }
  return true;
}

Maybe<void> InferSliceOpTensorDesc(user_op::InferContext* ctx) {
  const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const int64_t ndim = x_shape->NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim);

  const SbpParallel& out_sbp = ctx->SbpParallel4ArgNameAndIndex("y", 0);
  if (ctx->parallel_ctx().parallel_num() != 1 && out_sbp.has_split_parallel()) {
    FOR_RANGE(int, i, 0, ndim) {
      if (out_sbp.split_parallel().axis() == i) {
        CHECK_OR_RETURN(
            IsFullSlice(start_vec.at(i), stop_vec.at(i), step_vec.at(i), x_shape->At(i)));
      }
    }
  }

  DimVector dim_vec(ndim);
  FOR_RANGE(size_t, i, 0, dim_vec.size()) {
    const int64_t dim_size = x_shape->At(i);
    if (dim_size == 0) {
      dim_vec[i] = 0;
      continue;
    }
    const int64_t step = step_vec.at(i);
    CHECK_NE_OR_RETURN(step, 0) << "slice step cannot be 0";
    int64_t start = RegulateSliceStart(start_vec.at(i), dim_size);
    int64_t stop = RegulateSliceStop(stop_vec.at(i), dim_size);
    if (step > 0) {
      CHECK_LT_OR_RETURN(start, stop) << "slice start must be less than stop when step > 0"
                                         ", otherwise empty result will be outputted.";
    } else {
      CHECK_GT_OR_RETURN(start, stop) << "slice start must be more than stop when step < 0"
                                         ", otherwise empty result will be outputted.";
    }
    const int64_t diff = (step > 0) ? (stop - start - 1) : (stop - start + 1);
    dim_vec[i] = diff / step + 1;
  }
  *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(dim_vec);
  *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSliceOpSbpSignature(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const int64_t ndim = x_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim);

  FOR_RANGE(int, i, 0, ndim) {
    if (IsFullSlice(start_vec.at(i), stop_vec.at(i), step_vec.at(i), x_shape.At(i))) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> InferSliceGradOpTensorDesc(user_op::InferContext* ctx) {
  const Shape* like_shape = ctx->Shape4ArgNameAndIndex("like", 0);
  const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
  const int64_t ndim = dy_shape->NumAxes();
  CHECK_EQ_OR_RETURN(like_shape->NumAxes(), ndim);

  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim);

  const SbpParallel& dx_sbp = ctx->SbpParallel4ArgNameAndIndex("dx", 0);
  if (ctx->parallel_ctx().parallel_num() != 1 && dx_sbp.has_split_parallel()) {
    FOR_RANGE(int, i, 0, ndim) {
      if (dx_sbp.split_parallel().axis() == i) {
        CHECK_OR_RETURN(
            IsFullSlice(start_vec.at(i), stop_vec.at(i), step_vec.at(i), like_shape->At(i)));
      }
    }
  }

  *ctx->Shape4ArgNameAndIndex("dx", 0) = *like_shape;
  *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("dy", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetSliceGradOpSbpSignature(user_op::SbpContext* ctx) {
  const Shape& like_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape();
  const int64_t ndim = like_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim);

  FOR_RANGE(int, i, 0, ndim) {
    if (IsFullSlice(start_vec.at(i), stop_vec.at(i), step_vec.at(i), like_shape.At(i))) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
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
}

void InferSliceGradInputArgModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
                                    const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* dy_modifier = GetInputArgModifierFn("dy", 0);
  CHECK_NOTNULL(dy_modifier);
  dy_modifier->set_requires_grad(false);
  user_op::InputArgModifier* like_modifier = GetInputArgModifierFn("like", 0);
  CHECK_NOTNULL(like_modifier);
  like_modifier->set_requires_grad(false);
}

Maybe<void> InferSliceUpdateOpTensorDesc(user_op::InferContext* ctx) {
  const auto* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const int64_t ndim = x_desc->shape().NumAxes();
  const auto* update_desc = ctx->TensorDesc4ArgNameAndIndex("update", 0);
  CHECK_EQ_OR_RETURN(update_desc->shape().NumAxes(), ndim);
  CHECK_EQ_OR_RETURN(update_desc->data_type(), x_desc->data_type());

  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim);
  // validate update shape and start, stop, step attributes
  FOR_RANGE(int, i, 0, ndim) {
    const int64_t dim_size = x_desc->shape().At(i);
    const int64_t step = step_vec.at(i);
    CHECK_NE_OR_RETURN(step, 0) << "slice step cannot be 0";
    int64_t start = RegulateSliceStart(start_vec.at(i), dim_size);
    int64_t stop = RegulateSliceStop(stop_vec.at(i), dim_size);
    if (step > 0) {
      CHECK_LT_OR_RETURN(start, stop) << "slice start must be less than stop when step > 0"
                                         ", otherwise empty result will be outputted.";
    } else {
      CHECK_GT_OR_RETURN(start, stop) << "slice start must be more than stop when step < 0"
                                         ", otherwise empty result will be outputted.";
    }
    const int64_t diff = (step > 0) ? (stop - start - 1) : (stop - start + 1);
    const int64_t sliced_dim_size = diff / step + 1;
    CHECK_EQ_OR_RETURN(sliced_dim_size, update_desc->shape().At(i))
        << "sliced dim size " << sliced_dim_size << " at axis " << i
        << " not equal to the update shape " << update_desc->shape().ToString();
  }
  // the split axis can't be sliced
  const SbpParallel& x_sbp = ctx->SbpParallel4ArgNameAndIndex("x", 0);
  if (ctx->parallel_ctx().parallel_num() != 1 && x_sbp.has_split_parallel()) {
    const int64_t split_axis = x_sbp.split_parallel().axis();
    CHECK_GE_OR_RETURN(split_axis, 0);
    CHECK_LT_OR_RETURN(split_axis, ndim);
    CHECK_OR_RETURN(IsFullSlice(start_vec.at(split_axis), stop_vec.at(split_axis),
                                step_vec.at(split_axis), x_desc->shape().At(split_axis)));
  }

  auto* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  *y_desc = *x_desc;
  return Maybe<void>::Ok();
}

Maybe<void> GetSliceUpdateOpSbpSignature(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const int64_t ndim = x_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim);
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim);

  FOR_RANGE(int, i, 0, ndim) {
    if (IsFullSlice(start_vec.at(i), stop_vec.at(i), step_vec.at(i), x_shape.At(i))) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

void GenSliceGradOp(const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op = builder.Op("slice_grad")
                                             .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                             .Input("like", op.input("x", 0))
                                             .Attr("start", op.attr<std::vector<int64_t>>("start"))
                                             .Attr("stop", op.attr<std::vector<int64_t>>("stop"))
                                             .Attr("step", op.attr<std::vector<int64_t>>("step"))
                                             .Output("dx")
                                             .Build();
    op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
    AddOp(grad_op);
  }
}

Maybe<void> InferLogicalSliceAssignTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* ref_desc = ctx->TensorDesc4ArgNameAndIndex("ref", 0);
  user_op::TensorDesc* value_desc = ctx->TensorDesc4ArgNameAndIndex("value", 0);
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_OR_RETURN(!ref_desc->is_dynamic());
  CHECK_OR_RETURN(ref_desc->data_type() == value_desc->data_type());
  FOR_RANGE(size_t, i, 0, step_vec.size()) {
    const int64_t step = step_vec.at(i);
    const int64_t start = start_vec.at(i);
    const int64_t stop = stop_vec.at(i);
    CHECK_GT_OR_RETURN(step, 0) << "logical_slice_assign step must be greater than 0";
    CHECK_GE_OR_RETURN(start, 0) << "logical_slice_assign start must be greater or equal to 0";
    CHECK_GT_OR_RETURN(stop, 0) << "logical_slice_assign stop must be greater than 0";
    CHECK_LT_OR_RETURN(start, stop) << "logical_slice_assign start must be less than stop";
  }
  CHECK_OR_RETURN(ref_desc->is_tensor_list() == value_desc->is_tensor_list());
  return Maybe<void>::Ok();
}

Maybe<void> GetLogicalSliceAssignSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("ref", 0);
  FOR_RANGE(int64_t, axis, 0, ref_desc.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("ref", 0), axis)
        // TODO(jianhao): Support (S(n), S(n)) when axis n is not sliced
        .Broadcast(user_op::OpArg("value", 0))
        .Build();
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("ref", 0))
      .PartialSum(user_op::OpArg("value", 0))
      .Build();
  return Maybe<void>::Ok();
}

void InferLogicalSliceAssignInputArgModifier(user_op::GetInputArgModifier GetInputArgModifierFn,
                                             const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* ref_modifier = GetInputArgModifierFn("ref", 0);
  CHECK(ref_modifier != nullptr);
  ref_modifier->set_is_mutable(true);
  user_op::InputArgModifier* value_modifier = GetInputArgModifierFn("value", 0);
  CHECK(value_modifier != nullptr);
  value_modifier->set_requires_grad(false);
}

Maybe<void> InferLogicalSliceTensorDesc(user_op::InferContext* ctx) {
  const Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
  const int64_t ndim = x_shape->NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  DimVector dim_vec(ndim);
  FOR_RANGE(size_t, i, 0, dim_vec.size()) {
    const int64_t step = step_vec.at(i);
    const int64_t start = start_vec.at(i);
    const int64_t stop = stop_vec.at(i);
    CHECK_GT_OR_RETURN(step, 0) << "LogicalSlice step must be greater than 0";
    CHECK_GE_OR_RETURN(start, 0) << "LogicalSlice start must be greater or equal to 0";
    CHECK_GT_OR_RETURN(stop, 0) << "LogicalSlice stop must be greater than 0";
    CHECK_LT_OR_RETURN(start, stop) << "LogicalSlice start must be less than stop";
    const int64_t diff = stop - start - 1;
    dim_vec[i] = diff / step + 1;
  }
  *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(dim_vec);
  *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetLogicalSliceSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, axis, 0, input_desc.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), axis)
        // TODO(jianhao): Support S(n) -> S(n) when axis n is not sliced
        .PartialSum(user_op::OpArg("y", 0))
        .Build();
  }
  ctx->NewBuilder().PartialSum(user_op::OpArg("x", 0)).PartialSum(user_op::OpArg("y", 0)).Build();
  return Maybe<void>::Ok();
}

void GenSliceUpdateGradOp(user_op::BackwardOpConfContext* ctx) {
  const std::string update_grad_op_name = ctx->FwOp().op_name() + "_update_grad";
  ctx->DefineOp(update_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("slice")
        .InputBind("x", ctx->FwOp().output_grad("y", 0))
        .Attr("start", ctx->FwOp().attr<std::vector<int64_t>>("start"))
        .Attr("stop", ctx->FwOp().attr<std::vector<int64_t>>("stop"))
        .Attr("step", ctx->FwOp().attr<std::vector<int64_t>>("step"))
        .Output("y")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("update", 0), [&]() -> const std::string& {
    return ctx->GetOp(update_grad_op_name).output("y", 0);
  });

  const std::string zero_grad_op_name = ctx->FwOp().op_name() + "_zero_grad";
  ctx->DefineOp(zero_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("zero_like")
        .InputBind("like", ctx->FwOp().input("update", 0))
        .Output("out")
        .Build();
  });
  const std::string x_grad_op_name = ctx->FwOp().op_name() + "_x_grad";
  ctx->DefineOp(x_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("slice_update")
        .InputBind("x", ctx->FwOp().output_grad("y", 0))
        .InputBind("update", ctx->GetOp(zero_grad_op_name).output("out", 0))
        .Attr("start", ctx->FwOp().attr<std::vector<int64_t>>("start"))
        .Attr("stop", ctx->FwOp().attr<std::vector<int64_t>>("stop"))
        .Attr("step", ctx->FwOp().attr<std::vector<int64_t>>("step"))
        .Output("y")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), [&]() -> const std::string& {
    return ctx->GetOp(x_grad_op_name).output("y", 0);
  });
}

}  // namespace

REGISTER_USER_OP("slice")
    .Input("x")
    .Output("y")
    .Attr<std::vector<int64_t>>("start")
    .Attr<std::vector<int64_t>>("stop")
    .Attr<std::vector<int64_t>>("step")
    .SetTensorDescInferFn(InferSliceOpTensorDesc)
    .SetGetSbpFn(GetSliceOpSbpSignature);

REGISTER_USER_OP("slice_grad")
    .Input("dy")
    .Input("like")
    .Output("dx")
    .Attr<std::vector<int64_t>>("start")
    .Attr<std::vector<int64_t>>("stop")
    .Attr<std::vector<int64_t>>("step")
    .SetTensorDescInferFn(InferSliceGradOpTensorDesc)
    .SetGetSbpFn(GetSliceGradOpSbpSignature)
    .SetInputArgModifyFn(InferSliceGradInputArgModifier);

REGISTER_USER_OP("logical_slice_assign")
    .Input("ref")
    .Input("value")
    .Attr<std::vector<int64_t>>("start")
    .Attr<std::vector<int64_t>>("stop")
    .Attr<std::vector<int64_t>>("step")
    .SetTensorDescInferFn(InferLogicalSliceAssignTensorDesc)
    .SetGetSbpFn(GetLogicalSliceAssignSbpSignatures)
    .SetInputArgModifyFn(InferLogicalSliceAssignInputArgModifier);

REGISTER_USER_OP("logical_slice")
    .Input("x")
    .Output("y")
    .Attr<std::vector<int64_t>>("start")
    .Attr<std::vector<int64_t>>("stop")
    .Attr<std::vector<int64_t>>("step")
    .SetTensorDescInferFn(InferLogicalSliceTensorDesc)
    .SetGetSbpFn(GetLogicalSliceSbpSignatures);

REGISTER_USER_OP_GRAD("slice").SetGenBackwardOpConfFn(GenSliceGradOp);

REGISTER_USER_OP("slice_update")
    .Input("x")
    .Input("update")
    .Output("y")
    .Attr<std::vector<int64_t>>("start")
    .Attr<std::vector<int64_t>>("stop")
    .Attr<std::vector<int64_t>>("step")
    .SetTensorDescInferFn(InferSliceUpdateOpTensorDesc)
    .SetGetSbpFn(GetSliceUpdateOpSbpSignature);

REGISTER_USER_OP_GRAD("slice_update").SetBackwardOpConfGenFn(GenSliceUpdateGradOp);

}  // namespace oneflow
