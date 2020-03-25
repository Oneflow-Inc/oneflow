#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("top_k")
    .Input("in")
    .Output("out")
    .Attr("k", UserOpAttrType::kAtInt32)
    .Attr("sorted", UserOpAttrType::kAtBool)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      out_shape->Set(
          in_shape->NumAxes() - 1,
          std::min(ctx->GetAttr<int32_t>("k"), static_cast<int32_t>(in_shape->dim_vec().back())));
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // The current implementation can only do top_k in the last dimension and should use Broadcast
      // (by default) instead of Split for that dimension
      const int32_t num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
      if (num_axes > 1) {
        SbpSignatureBuilder()
            .Split(ctx->inputs(), 0)
            .Split(ctx->outputs(), 0)
            .MakeSplitSignatureListBuilder(num_axes - 1)
            .Build(ctx->sbp_sig_list());
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
