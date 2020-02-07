#include "oneflow/core/framework/batch_axis_context.h"

namespace oneflow {

namespace user_op {

Maybe<void> BatchAxisInferFnUtil::DefaultAsFirstHasValueInput(BatchAxisContext* ctx) {
  OptInt64* batch_axis = nullptr;
  for (const auto& pair : ctx->inputs()) {
    if (ctx->BatchAxis4ArgNameAndIndex(pair.first, pair.second)->has_value()) {
      batch_axis = ctx->BatchAxis4ArgNameAndIndex(pair.first, pair.second);
      break;
    }
  }
  if (batch_axis) {
    for (const auto& pair : ctx->outputs()) {
      *ctx->BatchAxis4ArgNameAndIndex(pair.first, pair.second) = *batch_axis;
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace user_op

}  // namespace oneflow
