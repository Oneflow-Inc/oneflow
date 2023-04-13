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
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferReduceDeviceStageDtypeFn(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  ctx->SetOutputDType("mask", 0, DataType::kBool);
  ctx->SetOutputDType("count", 0, DataType::kInt32);
  return Maybe<void>::Ok();
}

Maybe<void> InferReduceDeviceStageLogicalTensorDescFn(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
  const int64_t num_axes = input_shape.NumAxes();
  Shape output_shape;
  if (axis.empty()) {
    output_shape = Shape::Ones(num_axes);
  } else {
    const ParallelDesc& parallel_desc = ctx->parallel_desc();
    const NdSbp& in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
    DimVector dim_vec = input_shape.dim_vec();
    if (parallel_desc.hierarchy()->NumAxes() == 1) {
      const auto& input_sbp = in_nd_sbp.sbp_parallel(0);
      for (auto i : axis) {
        const int64_t regular_axis = ShiftNegativeAxis(i, num_axes);
        dim_vec.at(regular_axis) =
            (input_sbp.has_split_parallel() && input_sbp.split_parallel().axis() == regular_axis)
                ? parallel_desc.parallel_num()
                : 1;
      }
    } else {
      CHECK_EQ_OR_RETURN(axis.size(), 1);
      const int64_t regular_axis = ShiftNegativeAxis(axis.at(0), num_axes);
      dim_vec.at(regular_axis) = 1;
      for (int64_t i = 0; i < parallel_desc.hierarchy()->NumAxes(); ++i) {
        const auto& input_sbp = in_nd_sbp.sbp_parallel(i);
        if (input_sbp.has_split_parallel() && input_sbp.split_parallel().axis() == regular_axis) {
          dim_vec.at(regular_axis) *= parallel_desc.hierarchy()->At(i);
        }
      }
    }
    output_shape = Shape(dim_vec);
  }
  ctx->SetOutputShape("out", 0, output_shape);
  ctx->SetOutputShape("mask", 0, input_shape);
  ctx->SetOutputShape("count", 0, output_shape);

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceDeviceStagePhysicalTensorDescFn(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
  Shape output_shape;
  if (axis.empty()) {
    output_shape = Shape::Ones(input_shape.NumAxes());
  } else {
    const AxisVector axis_vec = {axis.begin(), axis.end()};
    const Shape& reduced_shape = CreateReducedShape(input_shape, axis_vec);
    output_shape = reduced_shape;
  }

  ctx->SetOutputShape("out", 0, output_shape);
  ctx->SetOutputShape("mask", 0, input_shape);
  ctx->SetOutputShape("count", 0, output_shape);
  ;

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceDeviceStageGradDtypeFn(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("mask", 0), DataType::kBool)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kBool) << ", but got "
      << DataType_Name(ctx->InputDType("mask", 0));
  CHECK_EQ_OR_RETURN(ctx->InputDType("count", 0), DataType::kInt32)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kInt32) << ", but got "
      << DataType_Name(ctx->InputDType("count", 0));
  ctx->SetOutputDType("in_diff", 0, ctx->InputDType("out_diff", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferReduceDeviceStageGradTensorDescFn(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputShape("out_diff", 0), ctx->InputShape("count", 0));
  ctx->SetOutputShape("in_diff", 0, ctx->InputShape("mask", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferReduceGlobalStageDtypeFn(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("device_count", 0), DataType::kInt32)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kInt32) << ", but got "
      << DataType_Name(ctx->InputDType("device_count", 0));
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  ctx->SetOutputDType("mask", 0, DataType::kBool);

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceGlobalStageTensorDescFn(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("in", 0);
  const Shape& device_count_shape = ctx->InputShape("device_count", 0);
  CHECK_EQ_OR_RETURN(input_shape, device_count_shape);
  const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
  bool keepdims = ctx->Attr<bool>("keepdims");
  Shape output_shape;
  if (axis.empty()) {
    if (keepdims) {
      output_shape = Shape::Ones(input_shape.NumAxes());
    } else {
      output_shape = Shape({1});
    }
  } else {
    const AxisVector axis_vec = {axis.begin(), axis.end()};
    const Shape& reduced_shape = CreateReducedShape(input_shape, axis_vec);
    if (keepdims) {
      output_shape = reduced_shape;
    } else {
      output_shape = reduced_shape.RemoveOnes(axis_vec);
    }
  }

  ctx->SetOutputShape("out", 0, output_shape);
  ctx->SetOutputShape("mask", 0, input_shape);

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceGlobalStageGradDtypeFn(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("mask", 0), DataType::kBool)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kBool) << ", but got "
      << DataType_Name(ctx->InputDType("mask", 0));
  CHECK_EQ_OR_RETURN(ctx->InputDType("device_count", 0), DataType::kInt32)
      << "InferDataType Failed. Expected " << DataType_Name(DataType::kInt32) << ", but got "
      << DataType_Name(ctx->InputDType("device_count", 0));

  ctx->SetOutputDType("in_diff", 0, ctx->InputDType("out_diff", 0));

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceGlobalStageGradTensorDescFn(user_op::InferContext* ctx) {
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  const Shape& device_count_shape = ctx->InputShape("device_count", 0);
  CHECK_EQ_OR_RETURN(device_count_shape, mask_shape);
  ctx->SetOutputShape("in_diff", 0, mask_shape);
  return Maybe<void>::Ok();
}

Maybe<void> GetReduceDeviceStageSbpFn(user_op::SbpContext* ctx) {
  int32_t num_axes = 0;
  HashSet<int32_t> conf_axes;
  {
    const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
    num_axes = in_tensor.shape().NumAxes();
    const auto& reduced_axes = ctx->Attr<std::vector<int32_t>>("axis");
    conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  }
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(conf_axes, num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), i)
        .Split(user_op::OpArg("out", 0), i)
        .Split(user_op::OpArg("mask", 0), i)
        .Split(user_op::OpArg("count", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetReduceDeviceStageGradSbpFn(user_op::SbpContext* ctx) {
  int32_t num_axes = 0;
  HashSet<int32_t> conf_axes;
  {
    const auto& output_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("out_diff", 0);
    num_axes = output_tensor.shape().NumAxes();
    const auto& reduced_axes = ctx->Attr<std::vector<int32_t>>("axis");
    conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  }
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(conf_axes, num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i) || i == 0) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("out_diff", 0), i)
          .Split(user_op::OpArg("count", 0), i)
          .Split(user_op::OpArg("mask", 0), i)
          .Split(user_op::OpArg("in_diff", 0), i)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

#define IMPLEMENT_REDUCE_DEVICE_STAGE_USER_OP_FUNCS(op_name)                                \
  /*static*/ Maybe<void> op_name##Op::GetSbp(user_op::SbpContext* ctx) {                    \
    return GetReduceDeviceStageSbpFn(ctx);                                                  \
  }                                                                                         \
  /*static*/ Maybe<void> op_name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferReduceDeviceStageLogicalTensorDescFn(ctx);                                  \
  }                                                                                         \
  /*static*/ Maybe<void> op_name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferReduceDeviceStagePhysicalTensorDescFn(ctx);                                 \
  }                                                                                         \
  /*static*/ Maybe<void> op_name##Op::InferDataType(user_op::InferContext* ctx) {           \
    return InferReduceDeviceStageDtypeFn(ctx);                                              \
  }

IMPLEMENT_REDUCE_DEVICE_STAGE_USER_OP_FUNCS(ReduceMinDeviceStage)
IMPLEMENT_REDUCE_DEVICE_STAGE_USER_OP_FUNCS(ReduceMaxDeviceStage)
#undef IMPLEMENT_REDUCE_DEVICE_STAGE_USER_OP_FUNCS

#define IMPLEMENT_REDUCE_DEVICE_STAGE_USER_GRAD_OP_FUNCS(op_name)                               \
  /*static*/ Maybe<void> op_name##GradOp::GetSbp(user_op::SbpContext* ctx) {                    \
    return GetReduceDeviceStageGradSbpFn(ctx);                                                  \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferReduceDeviceStageGradTensorDescFn(ctx);                                         \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                         \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferDataType(user_op::InferContext* ctx) {           \
    return InferReduceDeviceStageGradDtypeFn(ctx);                                              \
  }

IMPLEMENT_REDUCE_DEVICE_STAGE_USER_GRAD_OP_FUNCS(ReduceMinDeviceStage)
IMPLEMENT_REDUCE_DEVICE_STAGE_USER_GRAD_OP_FUNCS(ReduceMaxDeviceStage)
#undef IMPLEMENT_REDUCE_DEVICE_STAGE_USER_GRAD_OP_FUNCS

#define IMPLEMENT_REDUCE_GLOBAL_STAGE_OP_FUNCS(op_name)                                          \
  /*static*/ Maybe<void> op_name##Op::GetSbp(user_op::SbpContext* ctx) {                         \
    ctx->NewBuilder()                                                                            \
        .Split(user_op::OpArg("in", 0), 0)                                                       \
        .Split(user_op::OpArg("device_count", 0), 0)                                             \
        .Split(user_op::OpArg("out", 0), 0)                                                      \
        .Split(user_op::OpArg("mask", 0), 0)                                                     \
        .Build();                                                                                \
    return Maybe<void>::Ok();                                                                    \
  }                                                                                              \
  /*static*/ Maybe<void> op_name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {       \
    return InferReduceGlobalStageTensorDescFn(ctx);                                              \
  }                                                                                              \
  /*static*/ Maybe<void> op_name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) {      \
    return InferLogicalTensorDesc(ctx);                                                          \
  }                                                                                              \
  /*static*/ Maybe<void> op_name##Op::InferDataType(user_op::InferContext* ctx) {                \
    return InferReduceGlobalStageDtypeFn(ctx);                                                   \
  }                                                                                              \
  /*static*/ Maybe<void> op_name##Op::ModifyInputArg(                                            \
      const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {     \
    user_op::InputArgModifier* device_count_modifier = GetInputArgModifierFn("device_count", 0); \
    device_count_modifier->set_requires_grad(false);                                             \
    return Maybe<void>::Ok();                                                                    \
  }

IMPLEMENT_REDUCE_GLOBAL_STAGE_OP_FUNCS(ReduceMinGlobalStage)
IMPLEMENT_REDUCE_GLOBAL_STAGE_OP_FUNCS(ReduceMaxGlobalStage)
#undef IMPLEMENT_REDUCE_GLOBAL_STAGE_OP_FUNCS

#define IMPLEMENT_REDUCE_GLOBAL_STAGE_GRAD_OP_FUNCS(op_name)                                    \
  /*static*/ Maybe<void> op_name##GradOp::GetSbp(user_op::SbpContext* ctx) {                    \
    ctx->NewBuilder()                                                                           \
        .Split(user_op::OpArg("out_diff", 0), 0)                                                \
        .Split(user_op::OpArg("mask", 0), 0)                                                    \
        .Split(user_op::OpArg("device_count", 0), 0)                                            \
        .Split(user_op::OpArg("in_diff", 0), 0)                                                 \
        .Build();                                                                               \
    return Maybe<void>::Ok();                                                                   \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferReduceGlobalStageGradTensorDescFn(ctx);                                         \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                         \
  }                                                                                             \
  /*static*/ Maybe<void> op_name##GradOp::InferDataType(user_op::InferContext* ctx) {           \
    return InferReduceGlobalStageGradDtypeFn(ctx);                                              \
  }

IMPLEMENT_REDUCE_GLOBAL_STAGE_GRAD_OP_FUNCS(ReduceMinGlobalStage)
IMPLEMENT_REDUCE_GLOBAL_STAGE_GRAD_OP_FUNCS(ReduceMaxGlobalStage)
#undef IMPLEMENT_REDUCE_GLOBAL_STAGE_GRAD_OP_FUNCS

}  // namespace oneflow
