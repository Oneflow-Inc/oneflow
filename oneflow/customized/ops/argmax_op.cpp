#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("argmax")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      auto dim_vec = ctx->Shape4ArgNameAndIndex("in", 0)->dim_vec();
      dim_vec.pop_back();
      *ctx->Shape4ArgNameAndIndex("out", 0) =
          dim_vec.empty() ? Shape({1}) : Shape(std::move(dim_vec));
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      const Shape& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
      const auto* in_batch_axis = ctx->BatchAxis4ArgNameAndIndex("in", 0);
      if (in_batch_axis->has_value() && in_batch_axis->value() != in_shape.NumAxes() - 1) {
        *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *in_batch_axis;
      } else {
        ctx->BatchAxis4ArgNameAndIndex("out", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const Shape& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
      if (in_shape.NumAxes() > 1) {
        SbpSignatureBuilder()
            .Split(ctx->inputs(), 0)
            .Split(ctx->outputs(), 0)
            .MakeSplitSignatureListBuilder(in_shape.NumAxes() - 1)
            .Build(ctx->sbp_sig_list());
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
