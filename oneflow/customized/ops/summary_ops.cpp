#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_attr.pb.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("create_summary_writer")
    .Attr("logdir", UserOpAttrType::kAtString)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

REGISTER_USER_OP("write_scalar")
    .Input("in")
    .Input("step")
    .Input("tag")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      const Shape* step_shape = ctx->Shape4ArgNameAndIndex("step", 0);
      const Shape* tag_shape = ctx->Shape4ArgNameAndIndex("tag", 0);
      CHECK_OR_RETURN(in_shape->elem_cnt() == 1 && step_shape->elem_cnt() == 1);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

REGISTER_USER_OP("write_histogram")
    .Input("in")
    .Input("step")
    .Input("tag")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

REGISTER_USER_OP("write_text")
    .Input("in")
    .Input("step")
    .Input("tag")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

REGISTER_USER_OP("write_pb")
    .Input("in")
    .Input("step")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);

}  // namespace
}  // namespace oneflow