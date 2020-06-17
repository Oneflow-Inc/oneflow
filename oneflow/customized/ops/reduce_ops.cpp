#include "oneflow/core/framework/framework.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const Shape* input_shape = ctx->Shape4ArgNameAndIndex("input_tensor", 0);
  const auto& reduce_axes = ctx->Attr<std::vector<int32_t>>("axis");
  CHECK_OR_RETURN(!reduce_axes.empty());
  const AxisVector reduce_axes_vec = {reduce_axes.begin(), reduce_axes.end()};
  const Shape& reduce_shape = CreateReducedShape(*input_shape, reduce_axes_vec);
  const bool keepdims = ctx->Attr<bool>("keepdims");
  Shape* output_shape = ctx->Shape4ArgNameAndIndex("output_tensor", 0);
  if (keepdims) {
    *output_shape = reduce_shape;
  } else {
    *output_shape = reduce_shape.RemoveOnes(reduce_axes_vec);
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferBatchAxisFn(user_op::BatchAxisContext* ctx) {
  const auto& reduced_axes = ctx->Attr<std::vector<int32_t>>("axis");
  const bool keepdims = ctx->Attr<bool>("keepdims");
  HashSet<int32_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  const auto* in_batch_axis = ctx->BatchAxis4ArgNameAndIndex("input_tensor", 0);
  auto* out_batch_axis = ctx->BatchAxis4ArgNameAndIndex("output_tensor", 0);
  if (in_batch_axis->has_value()) {
    if (keepdims || conf_axes.find(in_batch_axis->value()) == conf_axes.end()) {
      *out_batch_axis = *in_batch_axis;
    } else {
      out_batch_axis->clear_value();
    }
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
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
}

template<template<typename> class binary_func>
Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input_tensor", 0);
  int64_t num_axes = in_tensor.shape().NumAxes();
  bool keep_dims = ctx->Attr<bool>("keepdims");
  const auto& reduce_axes = ctx->Attr<std::vector<int32_t>>("axis");
  HashSet<int32_t> conf_axes = {reduce_axes.begin(), reduce_axes.end()};
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

REGISTER_USER_OP_GRAD("reduce_sum")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("input_tensor", 0)) {
        const auto& axes = op.attr<std::vector<int32_t>>("axis");
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper reduce_sum_grad_op =
            builder.Op("broadcast_like")
                .Input("x", op.GetGradTensorWithOpOutput("output_tensor", 0))
                .Input("like", op.input("input_tensor", 0))
                .Attr("broadcast_axes", axes)
                .Output("y")
                .Build();
        op.BindGradTensorWithOpInput(reduce_sum_grad_op.output("y", 0), "input_tensor", 0);
        AddOp(reduce_sum_grad_op);
      }
    });

}  // namespace oneflow
