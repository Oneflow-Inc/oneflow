#include "oneflow/core/framework/framework.h"

namespace oneflow {

// TODO: deal with sbp and batch axis
REGISTER_USER_OP("top_k")
    .Input("in")
    .Output("out")
    .Attr("k", UserOpAttrType::kAtInt32)
    .Attr("sorted", UserOpAttrType::kAtBool)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      out_shape->dim_vec().back() = std::min(ctx->GetAttr("k"), in_shape.dim_vec().back());
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
