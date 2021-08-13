#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

REGISTER_NO_GRAD_USER_OP("send")
    .Input("in")
    .Attr<int64_t>("process_id")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // Do nothing.
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      UNIMPLEMENTED_THEN_RETURN();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // Do nothing.
      return Maybe<void>::Ok();
    });

}

}
