#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("clip_by_value")
    .Input("in")
    .OptionalInput("min")
    .OptionalInput("max")
    .Output("out")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* min_shape = ctx->Shape4ArgNameAndIndex("min", 0);
      Shape* max_shape = ctx->Shape4ArgNameAndIndex("max", 0);
      OF_CHECK_EQ(min_shape->NumAxes(), 1);
      OF_CHECK_EQ(max_shape->NumAxes(), 1);
      OF_CHECK_EQ(min_shape->At(0), 1);
      OF_CHECK_EQ(max_shape->At(0), 1);
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      DataType in_data_type = *ctx->Dtype4ArgNameAndIndex("in", 0);
      DataType min_data_type = *ctx->Dtype4ArgNameAndIndex("min", 0);
      DataType max_data_type = *ctx->Dtype4ArgNameAndIndex("max", 0);
      OF_CHECK_EQ(in_data_type, min_data_type);
      OF_CHECK_EQ(in_data_type, max_data_type);
      *ctx->Dtype4ArgNameAndIndex("out", 0) = in_data_type;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
      FOR_RANGE(int64_t, axis, 0, num_axes) {
        SbpSignatureBuilder()
            .Split("in", 0, axis)
            .Broadcast("min", 0)
            .Broadcast("max", 0)
            .Split("out", 0, axis)
            .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
