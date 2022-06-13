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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {
bool IsFullSlice(int64_t start, int64_t stop, int64_t step, int64_t size) {
  if (step != 1) { return false; }
  if (start != 0) { return false; }
  if (stop != size) { return false; }
  return true;
}
}  // namespace

/*static*/ Maybe<void> SliceUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("ref", 0).shape();
  const int64_t ndim = x_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  FOR_RANGE(int64_t, axis, 0, ndim) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("ref", 0), axis)
        .Broadcast(user_op::OpArg("value", 0))
        .Split(user_op::OpArg("y", 0), axis)
        .Build();
    // FullSlice support S+S->S
    if (IsFullSlice(start_vec[axis], stop_vec[axis], step_vec[axis], x_shape.At(axis))) {
      ctx->NewBuilder().Split(ctx->inputs(), axis).Split(ctx->outputs(), axis).Build();
    }
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("ref", 0))
      .PartialSum(user_op::OpArg("value", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->InputTensorDesc("ref", 0);
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_OR_RETURN(!ref_desc.is_dynamic());
  FOR_RANGE(size_t, i, 0, step_vec.size()) {
    const int64_t step = step_vec.at(i);
    const int64_t start = start_vec.at(i);
    const int64_t stop = stop_vec.at(i);
    CHECK_GT_OR_RETURN(step, 0) << "logical_slice_assign step must be greater than 0";
    CHECK_GE_OR_RETURN(start, 0) << "logical_slice_assign start must be greater or equal to 0";
    CHECK_GT_OR_RETURN(stop, 0) << "logical_slice_assign stop must be greater than 0";
    CHECK_LT_OR_RETURN(start, stop) << "logical_slice_assign start must be less than stop";
  }
  auto* y_desc = ctx->OutputTensorDesc("y", 0);
  *y_desc->mut_shape() = ref_desc.shape();
  *y_desc->mut_is_dynamic() = ref_desc.is_dynamic();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SliceUpdateOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->InputTensorDesc("ref", 0);
  const user_op::TensorDesc& value_desc = ctx->InputTensorDesc("value", 0);
  CHECK_OR_RETURN(ref_desc.data_type() == value_desc.data_type());
  auto* y_desc = ctx->OutputTensorDesc("y", 0);
  *y_desc->mut_data_type() = ref_desc.data_type();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SliceOp::GetSbp(user_op::SbpContext* ctx) {
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
/*static*/ Maybe<void> SliceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const int64_t ndim = x_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  DimVector dim_vec(ndim);
  FOR_RANGE(size_t, i, 0, dim_vec.size()) {
    const int64_t step = step_vec.at(i);
    const int64_t start = start_vec.at(i);
    const int64_t stop = stop_vec.at(i);
    CHECK_GT_OR_RETURN(step, 0) << "Slice step must be greater than 0";
    CHECK_GE_OR_RETURN(start, 0) << "Slice start must be greater or equal to 0";
    CHECK_GT_OR_RETURN(stop, 0) << "Slice stop must be greater than 0";
    CHECK_LT_OR_RETURN(start, stop) << "Slice start must be less than stop";
    const int64_t diff = stop - start - 1;
    dim_vec[i] = diff / step + 1;
  }
  *ctx->OutputShape("y", 0) = Shape(dim_vec);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SliceOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> GenSliceUpdateGradOp(user_op::BackwardOpConfContext* ctx) {
  const std::string update_grad_op_name = ctx->FwOp().op_name() + "_value_grad";
  ctx->DefineOp(update_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("logical_slice")
        .InputBind("x", ctx->FwOp().output_grad("y", 0))
        .Attr("start", ctx->FwOp().attr<std::vector<int64_t>>("start"))
        .Attr("stop", ctx->FwOp().attr<std::vector<int64_t>>("stop"))
        .Attr("step", ctx->FwOp().attr<std::vector<int64_t>>("step"))
        .Output("y")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("value", 0), [&]() -> const std::string& {
    return ctx->GetOp(update_grad_op_name).output("y", 0);
  });

  const std::string zero_grad_op_name = ctx->FwOp().op_name() + "_zero_grad";
  ctx->DefineOp(zero_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("zero_like")
        .InputBind("like", ctx->FwOp().input("value", 0))
        .Output("out")
        .Build();
  });
  const std::string x_grad_op_name = ctx->FwOp().op_name() + "_x_grad";
  ctx->DefineOp(x_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("logical_slice_assign")
        .InputBind("ref", ctx->FwOp().output_grad("y", 0))
        .InputBind("value", ctx->GetOp(zero_grad_op_name).output("out", 0))
        .Attr("start", ctx->FwOp().attr<std::vector<int64_t>>("start"))
        .Attr("stop", ctx->FwOp().attr<std::vector<int64_t>>("stop"))
        .Attr("step", ctx->FwOp().attr<std::vector<int64_t>>("step"))
        .Output("y")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("ref", 0), [&]() -> const std::string& {
    return ctx->GetOp(x_grad_op_name).output("y", 0);
  });
  return Maybe<void>::Ok();
}

Maybe<void> GenSliceGradOp(user_op::BackwardOpConfContext* ctx) {
  const std::string zero_grad_op_name = ctx->FwOp().op_name() + "_zero_grad";
  ctx->DefineOp(zero_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("zero_like")
        .InputBind("like", ctx->FwOp().input("x", 0))
        .Output("out")
        .Build();
  });
  const std::string x_grad_op_name = ctx->FwOp().op_name() + "_x_grad";
  ctx->DefineOp(x_grad_op_name, [&](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("logical_slice_assign")
        .InputBind("ref", ctx->GetOp(zero_grad_op_name).output("out", 0))
        .InputBind("value", ctx->FwOp().output_grad("y", 0))
        .Attr("start", ctx->FwOp().attr<std::vector<int64_t>>("start"))
        .Attr("stop", ctx->FwOp().attr<std::vector<int64_t>>("stop"))
        .Attr("step", ctx->FwOp().attr<std::vector<int64_t>>("step"))
        .Output("y")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), [&]() -> const std::string& {
    return ctx->GetOp(x_grad_op_name).output("y", 0);
  });

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP_GRAD("slice_update").SetBackwardOpConfGenFn(GenSliceUpdateGradOp);
REGISTER_USER_OP_GRAD("slice").SetBackwardOpConfGenFn(GenSliceGradOp);

}  // namespace oneflow
