#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("gather_nd")
    .Input("x")
    .Input("indices")
    .Output("y")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* x_shape = ctx->Shape4ArgNameAndIndex("x", 0);
      Shape* indices_shape = ctx->Shape4ArgNameAndIndex("indices", 0);
      int64_t segm_dim = indices_shape->At(indices_shape->NumAxes() - 1);
      OF_CHECK_LE(segm_dim, x_shape->NumAxes());
      DimVector y_shape_vec = indices_shape->dim_vec();
      y_shape_vec.pop_back();
      FOR_RANGE(int64_t, i, segm_dim, x_shape->NumAxes()) { y_shape_vec.push_back(x_shape->At(i)); }
      *ctx->Shape4ArgNameAndIndex("y", 0) = Shape(y_shape_vec);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // TODO
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
