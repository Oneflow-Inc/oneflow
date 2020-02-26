#include "oneflow/core/framework/framework.h"

namespace oneflow {

// TODO: deal with sbp and batch axis
REGISTER_USER_OP("arg_sort")
    .Input("in")
    .Output("out")
    .Attr("dir", UserOpAttrType::kAtString)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
