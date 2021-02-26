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

namespace oneflow {

namespace {

Maybe<void> InferReduceDeviceStageTensorDescFn(user_op::InferContext* ctx) {
  Shape* input_shape = ctx->Shape4ArgNameAndIndex("in", 0);
  const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
  Shape* output_shape = ctx->Shape4ArgNameAndIndex("out", 0);
  if (axis.empty()) {
    *output_shape = Shape::Ones(input_shape->NumAxes());
  } else {
    const AxisVector axis_vec = {axis.begin(), axis.end()};
    const Shape& reduced_shape = CreateReducedShape(*input_shape, axis_vec);
    *output_shape = reduced_shape;
  }

  *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
  *ctx->Shape4ArgNameAndIndex("mask", 0) = *input_shape;
  *ctx->Dtype4ArgNameAndIndex("mask", 0) = DataType::kInt8;

  *ctx->Shape4ArgNameAndIndex("count", 0) = *output_shape;
  *ctx->Dtype4ArgNameAndIndex("count", 0) = DataType::kInt32;

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceDeviceStageGradTensorDescFn(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(*ctx->Shape4ArgNameAndIndex("out_diff", 0),
                     *ctx->Shape4ArgNameAndIndex("count", 0));
  CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("mask", 0), DataType::kInt8);
  CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("count", 0), DataType::kInt32);
  *ctx->Shape4ArgNameAndIndex("in_diff", 0) = *ctx->Shape4ArgNameAndIndex("mask", 0);
  *ctx->Dtype4ArgNameAndIndex("in_diff", 0) = *ctx->Dtype4ArgNameAndIndex("out_diff", 0);

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceGlobalStageTensorDescFn(user_op::InferContext* ctx) {
  const Shape* input_shape = ctx->Shape4ArgNameAndIndex("in", 0);
  const Shape* device_count_shape = ctx->Shape4ArgNameAndIndex("device_count", 0);
  CHECK_EQ_OR_RETURN(*input_shape, *device_count_shape);
  CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("device_count", 0), DataType::kInt32);
  const auto& axis = ctx->Attr<std::vector<int32_t>>("axis");
  bool keepdims = ctx->Attr<bool>("keepdims");
  Shape* output_shape = ctx->Shape4ArgNameAndIndex("out", 0);
  if (axis.empty()) {
    if (keepdims) {
      *output_shape = Shape::Ones(input_shape->NumAxes());
    } else {
      *output_shape = Shape({1});
    }
  } else {
    const AxisVector axis_vec = {axis.begin(), axis.end()};
    const Shape& reduced_shape = CreateReducedShape(*input_shape, axis_vec);
    if (keepdims) {
      *output_shape = reduced_shape;
    } else {
      *output_shape = reduced_shape.RemoveOnes(axis_vec);
    }
  }

  *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
  *ctx->Shape4ArgNameAndIndex("mask", 0) = *input_shape;
  *ctx->Dtype4ArgNameAndIndex("mask", 0) = DataType::kInt8;

  return Maybe<void>::Ok();
}

Maybe<void> InferReduceGlobalStageGradTensorDescFn(user_op::InferContext* ctx) {
  Shape* mask_shape = ctx->Shape4ArgNameAndIndex("mask", 0);
  Shape* device_count_shape = ctx->Shape4ArgNameAndIndex("device_count", 0);
  CHECK_EQ_OR_RETURN(*device_count_shape, *mask_shape);

  CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("mask", 0), DataType::kInt8);
  CHECK_EQ_OR_RETURN(*ctx->Dtype4ArgNameAndIndex("device_count", 0), DataType::kInt32);

  *ctx->Shape4ArgNameAndIndex("in_diff", 0) = *mask_shape;
  *ctx->Dtype4ArgNameAndIndex("in_diff", 0) = *ctx->Dtype4ArgNameAndIndex("out_diff", 0);

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
    if (IsReducedAxis(i)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), i)
          .Split(user_op::OpArg("out", 0), i)
          .Split(user_op::OpArg("mask", 0), i)
          .Split(user_op::OpArg("count", 0), i)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetReduceDeviceStageGradSbpFn(user_op::SbpContext* ctx) {
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
    if (IsReducedAxis(i)) {
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

#define REGISTER_REDUCE_DEVICE_STAGE_USER_OP(op_name)           \
  REGISTER_USER_OP(op_name)                                     \
      .Input("in")                                              \
      .Output("out")                                            \
      .Output("mask")                                           \
      .Output("count")                                          \
      .Attr<std::vector<int32_t>>("axis")                       \
      .SetTensorDescInferFn(InferReduceDeviceStageTensorDescFn) \
      .SetGetSbpFn(GetReduceDeviceStageSbpFn);

REGISTER_REDUCE_DEVICE_STAGE_USER_OP("reduce_min_device_stage")
REGISTER_REDUCE_DEVICE_STAGE_USER_OP("reduce_max_device_stage")

#define REGISTER_REDUCE_DEVICE_STAGE_GRAD_USER_OP(op_name)          \
  REGISTER_USER_OP(op_name)                                         \
      .Input("out_diff")                                            \
      .Input("mask")                                                \
      .Input("count")                                               \
      .Output("in_diff")                                            \
      .Attr<std::vector<int32_t>>("axis")                           \
      .SetTensorDescInferFn(InferReduceDeviceStageGradTensorDescFn) \
      .SetGetSbpFn(GetReduceDeviceStageGradSbpFn);

REGISTER_REDUCE_DEVICE_STAGE_GRAD_USER_OP("reduce_min_device_stage_grad")
REGISTER_REDUCE_DEVICE_STAGE_GRAD_USER_OP("reduce_max_device_stage_grad")

void GenBackwardOpConf4ReduceDeviceStage(const std::string& op_type_name,
                                         const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op =
        builder.Op(op_type_name)
            .Input("mask", op.output("mask", 0))
            .Input("count", op.output("count", 0))
            .Input("out_diff", op.GetGradTensorWithOpOutput("out", 0))
            .Output("in_diff")
            .Attr("axis", op.attr<std::vector<int32_t>>("axis"))
            .Build();
    op.BindGradTensorWithOpInput(grad_op.output("in_diff", 0), "in", 0);
    AddOp(grad_op);
  }
}

#define REGISTER_REDUCE_DEVICE_STAGE_USER_OP_GRAD(op_type_name, grad_op_type_name)           \
  REGISTER_USER_OP_GRAD(op_type_name)                                                        \
      .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) { \
        return GenBackwardOpConf4ReduceDeviceStage(grad_op_type_name, op, AddOp);            \
      });
REGISTER_REDUCE_DEVICE_STAGE_USER_OP_GRAD("reduce_min_device_stage", "reduce_min_device_stage_grad")
REGISTER_REDUCE_DEVICE_STAGE_USER_OP_GRAD("reduce_max_device_stage", "reduce_max_device_stage_grad")

#define REGISTER_REDUCE_GLOBAL_STAGE_USER_OP(op_name)                             \
  REGISTER_USER_OP(op_name)                                                       \
      .Input("in")                                                                \
      .Input("device_count")                                                      \
      .Output("out")                                                              \
      .Output("mask")                                                             \
      .Attr<std::vector<int32_t>>("axis")                                         \
      .Attr<bool>("keepdims")                                                     \
      .SetTensorDescInferFn(InferReduceGlobalStageTensorDescFn)                   \
      .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn, \
                              const user_op::UserOpConfWrapper&) {                \
        user_op::InputArgModifier* device_count_modifier =                        \
            GetInputArgModifierFn("device_count", 0);                             \
        device_count_modifier->set_requires_grad(false);                          \
      })                                                                          \
      .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); });

REGISTER_REDUCE_GLOBAL_STAGE_USER_OP("reduce_min_global_stage")
REGISTER_REDUCE_GLOBAL_STAGE_USER_OP("reduce_max_global_stage")

#define REGISTER_REDUCE_GLOBAL_STAGE_GRAD_USER_OP(op_name)          \
  REGISTER_USER_OP(op_name)                                         \
      .Input("out_diff")                                            \
      .Input("mask")                                                \
      .Input("device_count")                                        \
      .Output("in_diff")                                            \
      .Attr<std::vector<int32_t>>("axis")                           \
      .Attr<bool>("keepdims")                                       \
      .SetTensorDescInferFn(InferReduceGlobalStageGradTensorDescFn) \
      .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> { return Maybe<void>::Ok(); });

REGISTER_REDUCE_GLOBAL_STAGE_GRAD_USER_OP("reduce_min_global_stage_grad")
REGISTER_REDUCE_GLOBAL_STAGE_GRAD_USER_OP("reduce_max_global_stage_grad")

void GenBackwardOpConf4ReduceGlobalStage(const std::string& op_type_name,
                                         const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper grad_op =
        builder.Op(op_type_name)
            .Input("mask", op.output("mask", 0))
            .Input("device_count", op.input("device_count", 0))
            .Input("out_diff", op.GetGradTensorWithOpOutput("out", 0))
            .Output("in_diff")
            .Attr("axis", op.attr<std::vector<int32_t>>("axis"))
            .Attr("keepdims", op.attr<bool>("keepdims"))
            .Build();
    op.BindGradTensorWithOpInput(grad_op.output("in_diff", 0), "in", 0);
    AddOp(grad_op);
  }
}

#define REGISTER_REDUCE_GLOBAL_STAGE_USER_OP_GRAD(op_type_name, grad_op_type_name)           \
  REGISTER_USER_OP_GRAD(op_type_name)                                                        \
      .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) { \
        return GenBackwardOpConf4ReduceGlobalStage(grad_op_type_name, op, AddOp);            \
      });
REGISTER_REDUCE_GLOBAL_STAGE_USER_OP_GRAD("reduce_min_global_stage", "reduce_min_global_stage_grad")
REGISTER_REDUCE_GLOBAL_STAGE_USER_OP_GRAD("reduce_max_global_stage", "reduce_max_global_stage_grad")

}  // namespace oneflow
