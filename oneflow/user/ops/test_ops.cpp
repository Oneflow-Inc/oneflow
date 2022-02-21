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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> CcreluOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CcreluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("x", 0);
  Shape* out_shape = ctx->OutputShape("y", 0);
  *out_shape = in_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CcreluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> CcreluOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CcreluGradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("y", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CcreluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  Shape* dx_shape = ctx->OutputShape("dx", 0);
  CHECK_OR_RETURN(dy_shape == y_shape);
  *dx_shape = y_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CcreluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> CcreluGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("dx", 0) = ctx->InputDType("y", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("ccrelu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper ccrelu_grad_op =
        builder.Op("ccrelu_grad")
            .Input("y", op.output("y", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
            .Output("dx")
            .Build();
    op.BindGradTensorWithOpInput(ccrelu_grad_op.output("dx", 0), "x", 0);
    AddOp(ccrelu_grad_op);
  }
  return Maybe<void>::Ok();
});

/*static*/ Maybe<void> TestReshapeOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TestReshapeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  Shape* out_shape = ctx->OutputShape("out", 0);
  const Shape& conf_shape = ctx->Attr<Shape>("shape");
  CHECK_EQ_OR_RETURN(in_shape.NumAxes(), conf_shape.NumAxes());
  *out_shape = conf_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestReshapeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestReshapeOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestSourceOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestSourceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  Shape* out_shape = ctx->OutputShape("out", 0);
  *out_shape = Shape({5});
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestSourceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestSourceOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = DataType::kFloat;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestMultiOutputOrderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestMultiOutputOrderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  Shape* out1_shape = ctx->OutputShape("out1", 0);
  Shape* out2_shape = ctx->OutputShape("out2", 0);
  *out1_shape = in_shape;
  *out2_shape = in_shape;
  int32_t last_axis = in_shape.NumAxes() - 1;
  out2_shape->Set(last_axis, in_shape.At(last_axis) * 2);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestMultiOutputOrderOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestMultiOutputOrderOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out1", 0) = ctx->InputDType("in", 0);
  *ctx->OutputDType("out2", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestSourceMultiGpuFixedOutNumOp::GetSbp(user_op::SbpContext* ctx) {
  int64_t parallel_num = ctx->parallel_num();
  DeviceType device_type = ctx->device_type();
  if (device_type == DeviceType::kCPU && parallel_num > 1) {
    ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestSourceMultiGpuFixedOutNumOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  Shape* out_shape = ctx->OutputShape("out", 0);
  int64_t out_num = ctx->Attr<int64_t>("out_num");
  *out_shape = Shape({out_num});
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestSourceMultiGpuFixedOutNumOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  Shape* out_shape = ctx->OutputShape("out", 0);
  int64_t out_num = ctx->Attr<int64_t>("out_num");
  const ParallelContext& parallel_ctx = ctx->parallel_ctx();
  BalancedSplitter bs(out_num, parallel_ctx.parallel_num());
  *out_shape = Shape({bs.At(parallel_ctx.parallel_id()).size()});

  const SbpParallel& out_sbp = ctx->SbpParallel4ArgNameAndIndex("out", 0);
  CHECK_OR_RETURN(out_sbp.has_split_parallel() && out_sbp.split_parallel().axis() == 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestSourceMultiGpuFixedOutNumOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = DataType::kFloat;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestMultiInputOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x1_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x1", 0);
  FOR_RANGE(int64_t, i, 0, x1_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestMultiInputOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x1_shape = ctx->InputShape("x1", 0);
  const Shape& x2_shape = ctx->InputShape("x2", 0);
  Shape* y_shape = ctx->OutputShape("y", 0);
  CHECK_OR_RETURN(x1_shape == x2_shape);
  *y_shape = x1_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestMultiInputOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestMultiInputOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x1", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestMultiInputGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x1_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x1", 0);
  FOR_RANGE(int64_t, i, 0, x1_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestMultiInputGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x1_shape = ctx->InputShape("x1", 0);
  const Shape& x2_shape = ctx->InputShape("x2", 0);
  Shape* x1_diff_shape = ctx->OutputShape("x1_diff", 0);
  Shape* x2_diff_shape = ctx->OutputShape("x2_diff", 0);
  *x1_diff_shape = x1_shape;
  *x2_diff_shape = x2_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestMultiInputGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestMultiInputGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("x1_diff", 0) = ctx->InputDType("x1", 0);
  *ctx->OutputDType("x2_diff", 0) = ctx->InputDType("x2", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("TestMultiInput")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x1", 0) || op.NeedGenGradTensor4OpInput("x2", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper test_multi_input_grad_op =
            builder.Op("TestMultiInputGrad")
                .Input("x1", op.input("x1", 0))
                .Input("x2", op.input("x2", 0))
                .Input("y_diff", op.GetGradTensorWithOpOutput("y", 0))
                .Output("x1_diff")
                .Output("x2_diff")
                .Build();
        op.BindGradTensorWithOpInput(test_multi_input_grad_op.output("x1_diff", 0), "x1", 0);
        op.BindGradTensorWithOpInput(test_multi_input_grad_op.output("x2_diff", 0), "x2", 0);
        AddOp(test_multi_input_grad_op);
      }
      return Maybe<void>::Ok();
    });

/*static*/ Maybe<void> TestDynamicSourceOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestDynamicSourceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
  *out_tensor->mut_shape() = Shape({5});
  out_tensor->set_is_dynamic(true);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestDynamicSourceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestDynamicSourceOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = DataType::kFloat;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestDynamicSourceOp::ModifyOutputArg(
    const GetOutputArgModifier& GetOutputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::OutputArgModifier* out_modifier = GetOutputArgModifierFn("out", 0);
  CHECK_OR_RETURN(out_modifier != nullptr);
  out_modifier->set_header_infered_before_compute(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestRandomSourceOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TestRandomSourceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
  *out_tensor->mut_shape() = Shape({5});
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestRandomSourceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestRandomSourceOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = DataType::kFloat;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestDataTypeAttrOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TestDataTypeAttrOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("in", 0);
  Shape* out_shape = ctx->OutputShape("out", 0);
  *out_shape = in_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestDataTypeAttrOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestDataTypeAttrOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->Attr<DataType>("output_type");
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestListDataTypeAndListShapeAndListStringAttrOp::GetSbp(
    user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TestListDataTypeAndListShapeAndListStringAttrOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const auto& out_shapes = ctx->Attr<std::vector<Shape>>("out_shapes");
  const auto& string_list = ctx->Attr<std::vector<std::string>>("string_list");
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) {
    *ctx->OutputShape("out", i) = out_shapes.at(i);
  }
  CHECK_GT_OR_RETURN(string_list.size(), 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TestListDataTypeAndListShapeAndListStringAttrOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestListDataTypeAndListShapeAndListStringAttrOp::InferDataType(
    user_op::InferContext* ctx) {
  const auto& out_types = ctx->Attr<std::vector<DataType>>("out_types");
  FOR_RANGE(int32_t, i, 0, ctx->outputs().size()) { *ctx->OutputDType("out", i) = out_types.at(i); }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TestUserOpAttrAutoTypeOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}
/*static*/ Maybe<void> TestUserOpAttrAutoTypeOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::Unchanged(ctx);
}
/*static*/ Maybe<void> TestUserOpAttrAutoTypeOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TestUserOpAttrAutoTypeOp::InferDataType(user_op::InferContext* ctx) {
  return user_op::TensorDescInferFnUtil::UnchangedDataType(ctx);
}

/*static*/ Maybe<void> CpuOnlyReluTestOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CpuOnlyReluTestOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& in_desc = ctx->InputTensorDesc("in", 0);
  auto* out_desc = ctx->OutputTensorDesc("out", 0);
  *out_desc->mut_shape() = in_desc.shape();
  *out_desc->mut_is_dynamic() = in_desc.is_dynamic();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> CpuOnlyReluTestOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> CpuOnlyReluTestOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
