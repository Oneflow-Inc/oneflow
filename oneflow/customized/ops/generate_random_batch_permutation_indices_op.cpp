#include <cstdint>
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("generate_random_batch_permutation_indices")
    .Input("x")
    .Output("y")
    .Attr("seed", UserOpAttrType::kAtInt64)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("y", 0) = Shape({ctx->Shape4ArgNameAndIndex("x", 0)->At(0)});
      *ctx->Dtype4ArgNameAndIndex("y", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      if (ctx->BatchAxis4ArgNameAndIndex("x", 0)->has_value()
          && ctx->BatchAxis4ArgNameAndIndex("x", 0)->value() == 0) {
        ctx->BatchAxis4ArgNameAndIndex("y", 0)->set_value(0);
      } else {
        ctx->BatchAxis4ArgNameAndIndex("y", 0)->clear_value();
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      SbpSignatureBuilder().PartialSum("x").Broadcast("y").Build(
          ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      const int32_t num_axes =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
      FOR_RANGE(int64_t, i, 1, num_axes) {
        SbpSignatureBuilder().Split("x", i).Broadcast("y").Build(
            ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      }
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      GetInputArgModifierFn("x", 0)->set_use_header_only(true);
    });

}  // namespace oneflow
