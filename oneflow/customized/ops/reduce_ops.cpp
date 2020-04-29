#include "oneflow/core/framework/framework.h"

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
  const bool keep_dims = ctx->GetAttr<bool>("keepdims");
  HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  if (ctx->BatchAxis4ArgNameAndIndex("input_tensor", 0)->has_value() && !conf_axes.empty()
      && conf_axes.find(ctx->BatchAxis4ArgNameAndIndex("input_tensor", 0)->has_value())
             == conf_axes.end()) {
    *ctx->BatchAxis4ArgNameAndIndex("output_tensor", 0) =
        *ctx->BatchAxis4ArgNameAndIndex("input_tensor", 0);
  } else if (conf_axes.empty() && keep_dims == false) {
    ctx->BatchAxis4ArgNameAndIndex("output_tensor", 0)->set_value(0);
  } else {
    ctx->BatchAxis4ArgNameAndIndex("output_tensor", 0)->clear_value();
  }
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& input_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("input_tensor", 0);
  SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .MakeSplitSignatureListBuilder(input_tensor.shape().NumAxes())
      .Build(ctx->sbp_sig_list());
  return Maybe<void>::Ok();
}

#define REGISTER_REDUCE_USER_OP(op_name)          \
  REGISTER_USER_OP(op_name)                       \
      .Input("input_tensor")                      \
      .Output("output_tensor")                    \
      .Attr("axis", UserOpAttrType::kAtListInt32) \
      .Attr("keepdims", UserOpAttrType::kAtBool)  \
      .SetTensorDescInferFn(InferTensorDescFn)    \
      .SetBatchAxisInferFn(InferBatchAxisFn)      \
      .SetGetSbpFn(GetSbpFn);

REGISTER_REDUCE_USER_OP("reduce_any")
REGISTER_REDUCE_USER_OP("reduce_min")
REGISTER_REDUCE_USER_OP("reduce_prod")
REGISTER_REDUCE_USER_OP("reduce_all")
REGISTER_REDUCE_USER_OP("reduce_sum")

}  // namespace oneflow