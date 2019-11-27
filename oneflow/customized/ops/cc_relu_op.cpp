#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

REGISTER_USER_OP("ccrelu").Input("in").Output("out").SetShapeInferFn(
    [](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      int32_t last_axis = in_shape->NumAxes() - 1;
      out_shape->Set(last_axis, in_shape->At(last_axis) * 2);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
