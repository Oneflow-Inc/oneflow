#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  Shape* input_shape = ctx->Shape4ArgNameAndIndex("input_tensor", 0);
  const auto& axis = ctx->GetAttr<std::vector<int32_t>>("axis");
  bool keepdims = ctx->GetAttr<bool>("keepdims");
  Shape* output_shape = ctx->Shape4ArgNameAndIndex("output_tensor", 0);
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
  return Maybe<void>::Ok();
}

Maybe<void> InferBatchAxisFn(user_op::BatchAxisContext* ctx) {
  const auto& reduced_axes = ctx->GetAttr<std::vector<int32_t>>("axis");
  HashSet<int32_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  const auto* in_batch_axis = ctx->BatchAxis4ArgNameAndIndex("input_tensor", 0);
  auto* out_batch_axis = ctx->BatchAxis4ArgNameAndIndex("output_tensor", 0);
  if (in_batch_axis->has_value() && !conf_axes.empty()
      && conf_axes.find(in_batch_axis->value()) == conf_axes.end()) {
    *out_batch_axis = *in_batch_axis;
  } else {
    out_batch_axis->clear_value();
  }
  return Maybe<void>::Ok();
}

template<template<typename> class binary_func>
void GeneratePartialSbp(user_op::SbpContext* ctx, int64_t axis) {
  // TODO(lixinqi)
}

template<>
void GeneratePartialSbp<BinaryFuncSum>(user_op::SbpContext* ctx, int64_t axis) {
  ctx->NewBuilder().Split(ctx->inputs(), axis).PartialSum(ctx->outputs()).Build();
}

template<template<typename> class binary_func>
Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  int32_t num_axes = 0;
  bool keep_dims = false;
  HashSet<int32_t> conf_axes;
  {
    const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input_tensor", 0);
    num_axes = in_tensor.shape().NumAxes();
    keep_dims = ctx->GetAttr<bool>("keepdims");
    const auto& reduced_axes = ctx->GetAttr<std::vector<int32_t>>("axis");
    conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  }
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(conf_axes, num_axes);
  int32_t num_reduced_axes = 0;
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      GeneratePartialSbp<binary_func>(ctx, i);
      num_reduced_axes += 1;
    } else {
      ctx->NewBuilder()
          .Split(ctx->inputs(), i)
          .Split(ctx->outputs(), keep_dims ? i : i - num_reduced_axes)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

#define REGISTER_REDUCE_USER_OP(op_name, binary_func) \
  REGISTER_USER_OP(op_name)                           \
      .Input("input_tensor")                          \
      .Output("output_tensor")                        \
      .Attr("axis", UserOpAttrType::kAtListInt32)     \
      .Attr("keepdims", UserOpAttrType::kAtBool)      \
      .SetTensorDescInferFn(InferTensorDescFn)        \
      .SetBatchAxisInferFn(InferBatchAxisFn)          \
      .SetGetSbpFn(GetSbpFn<binary_func>);

REGISTER_REDUCE_USER_OP("reduce_any", BinaryFuncAny)
REGISTER_REDUCE_USER_OP("reduce_all", BinaryFuncAll)
REGISTER_REDUCE_USER_OP("reduce_min", BinaryFuncMin)
REGISTER_REDUCE_USER_OP("reduce_prod", BinaryFuncProd)
REGISTER_REDUCE_USER_OP("reduce_sum", BinaryFuncSum)

}  // namespace oneflow
