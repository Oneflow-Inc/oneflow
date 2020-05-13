#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

REGISTER_USER_OP("constant_like")
    .Input("like")
    .Attr("integer_value", UserOpAttrType::kAtInt64)
    .Attr("floating_value", UserOpAttrType::kAtDouble)
    .Attr("is_floating_value", UserOpAttrType::kAtBool)
    .Attr("dtype", UserOpAttrType::kAtDataType)
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      *ctx->Dtype4ArgNameAndIndex("out", 0) = ctx->GetAttr<DataType>("dtype");

      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      if (ctx->outputs().empty()) { return Maybe<void>::Ok(); }
      CHECK_GT_OR_RETURN(ctx->inputs().size(), 0);
      CHECK_EQ_OR_RETURN(ctx->outputs().size(), 1);
      const OptInt64* batch_axis = nullptr;
      for (const auto& ibn : ctx->inputs()) {
        const OptInt64* const cur_ibn_batch_axis =
            ctx->BatchAxis4ArgNameAndIndex(ibn.first, ibn.second);
        if (cur_ibn_batch_axis->has_value() == false) { continue; }
        if (batch_axis) {
          CHECK_OR_RETURN(*batch_axis == *cur_ibn_batch_axis);
        } else {
          batch_axis = cur_ibn_batch_axis;
        }
      }
      OptInt64 no_batch_axis;
      if (batch_axis == nullptr) { batch_axis = &no_batch_axis; }
      const auto& sole_out_arg_pair = ctx->outputs().at(0);
      *ctx->BatchAxis4ArgNameAndIndex(sole_out_arg_pair.first, sole_out_arg_pair.second) =
          *batch_axis;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      FOR_RANGE(int64_t, i, 0,
                ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("like", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("like", 0))
          .Broadcast(user_op::OpArg("out", 0))
          .Build();

      return Maybe<void>::Ok();
    });

}  // namespace oneflow