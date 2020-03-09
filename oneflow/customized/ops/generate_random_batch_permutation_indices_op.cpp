#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("generate_random_batch_permutation_indices")
    .Input("x")
    .Output("y")
    .Attr("seed", UserOpAttrType::kAtInt64)
    .Attr("has_seed", UserOpAttrType::kAtInt32)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Shape4ArgNameAndIndex("y", 0) = Shape({ctx->Shape4ArgNameAndIndex("x", 0)->At(0)});
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("y", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      *ctx->BatchAxis4ArgNameAndIndex("y", 0) = *ctx->BatchAxis4ArgNameAndIndex("x", 0);
      return Maybe<void>::Ok();
    })
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn) {
      GetInputArgModifierFn("x", 0)->set_use_header_only(true);
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
