#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("random_like")
    .Input("like")
    .Output("out")
    .Attr("seed", UserOpAttrType::kAtInt64)
    .Attr("has_seed", UserOpAttrType::kAtInt32)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("out", 0) = *ctx->Shape4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kFloat;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("out", 0) = *ctx->BatchAxis4ArgNameAndIndex("like", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const int32_t num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("like", 0).shape().NumAxes();
      SbpSignatureBuilder().MakeSplitSignatureListBuilder(num_axes).Build(ctx->sbp_sig_list());
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      GetInputArgModifierFn("like", 0)->set_use_header_only(true);
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
