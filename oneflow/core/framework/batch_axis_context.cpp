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

Maybe<void> BatchAxisInferFnUtil::NaiveInferBatchAxis(BatchAxisContext* ctx) {
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
  *ctx->BatchAxis4ArgNameAndIndex(sole_out_arg_pair.first, sole_out_arg_pair.second) = *batch_axis;
  return Maybe<void>::Ok();
}

}  // namespace user_op

}  // namespace oneflow
