#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_attr.pb.h"

namespace oneflow {

REGISTER_USER_OP("constant_like")
    .Input("like")
    .Attr("integer_value", UserOpAttrType::kAtInt64)
    .Attr("floating_value", UserOpAttrType::kAtDouble)
    .Attr("is_floating_value", UserOpAttrType::kAtBool)
    .Attr("dtype", UserOpAttrType::kAtDataType)
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("like", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      const auto& dtype = ctx->GetAttr<DataType>("data_type");
      *ctx->Dtype4ArgNameAndIndex("out", 0) = dtype;

      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      if (ctx->outputs().empty()) { return Maybe<void>::Ok(); }
      OF_CHECK_GT(ctx->inputs().size(), 0);
      OF_CHECK_EQ(ctx->outputs().size(), 1);
      const OptInt64* batch_axis = nullptr;
      for (const auto& in_arg_pair : ctx->inputs()) {
        const OptInt64* const cur_in_batch_axis =
            ctx->BatchAxis4ArgNameAndIndex(in_arg_pair.first, in_arg_pair.second);
        if (cur_in_batch_axis->has_value() == false) { continue; }
        if (batch_axis) {
          OF_CHECK_EQ(batch_axis->value(), cur_in_batch_axis->value());
        } else {
          batch_axis = cur_in_batch_axis;
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
      SbpSignatureBuilder()
          .Split("like", 0, 0)
          .Split("out", 0, 0)
          .MakeSplitSignatureListBuilder(
              ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes())
          .Build(ctx->sbp_sig_list());
      SbpSignatureBuilder().PartialSum("like", 0).Broadcast("out", 0).Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

}  // namespace oneflow