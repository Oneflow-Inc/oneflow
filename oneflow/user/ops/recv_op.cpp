#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

REGISTER_NO_GRAD_USER_OP("recv")
    .Output("out")
    .Attr<DataType>("dtype")
    .Attr<Shape>("shape")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->Attr<Shape>("shape");
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      UNIMPLEMENTED_THEN_RETURN();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->Attr<DataType>("dtype");
      return Maybe<void>::Ok();
    });

}
}
