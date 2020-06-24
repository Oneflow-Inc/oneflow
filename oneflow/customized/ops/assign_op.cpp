#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* ref_desc = ctx->TensorDesc4ArgNameAndIndex("ref", 0);
  user_op::TensorDesc* value_desc = ctx->TensorDesc4ArgNameAndIndex("value", 0);
  CHECK_OR_RETURN(!ref_desc->is_dynamic());
  CHECK_OR_RETURN(ref_desc->shape() == value_desc->shape());
  CHECK_OR_RETURN(ref_desc->data_type() == value_desc->data_type());
  CHECK_OR_RETURN(ref_desc->is_tensor_list() == value_desc->is_tensor_list());
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpSignatures(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("ref", 0);
  FOR_RANGE(int64_t, axis, 0, ref_desc.shape().NumAxes()) {
    ctx->NewBuilder().Split(ctx->inputs(), axis).Build();
  }
  return Maybe<void>::Ok();
}

void InputArgModifierFn(user_op::GetInputArgModifier GetInputArgModifierFn,
                        const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* ref_modifier = GetInputArgModifierFn("ref", 0);
  CHECK(ref_modifier != nullptr);
  ref_modifier->set_is_mutable(true);
  user_op::InputArgModifier* value_modifier = GetInputArgModifierFn("value", 0);
  CHECK(value_modifier != nullptr);
  value_modifier->set_requires_grad(false);
}

}  // namespace

REGISTER_USER_OP("assign")
    .Input("ref")
    .Input("value")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetGetSbpFn(GetSbpSignatures)
    .SetInputArgModifyFn(InputArgModifierFn);

}  // namespace oneflow
