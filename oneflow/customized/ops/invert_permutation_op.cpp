#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("invert_permutation")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* x_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* y_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *y_shape = *x_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

}  // namespace oneflow