#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

inline void SplitImpl(SbpSignature* sbp_sign, const std::string& bn, int64_t axis) {
  (*sbp_sign->mutable_bn_in_op2sbp_parallel())[bn].mutable_split_parallel()->set_axis(axis);
}

inline void BroadcastImpl(SbpSignature* sbp_sign, const std::string& bn) {
  (*sbp_sign->mutable_bn_in_op2sbp_parallel())[bn].mutable_broadcast_parallel();
}

inline void PartialSumImpl(SbpSignature* sbp_sign, const std::string& bn) {
  (*sbp_sign->mutable_bn_in_op2sbp_parallel())[bn].mutable_partial_sum_parallel();
}

}  // namespace

namespace user_op {

UserOpSbpSignatureBuilder&& UserOpSbpSignatureBuilder::Split(const OpArg& op_arg, int64_t axis) {
  SplitImpl(sbp_sign_, GenRepeatedBn(op_arg.name(), op_arg.index()), axis);
  return std::move(*this);
}

UserOpSbpSignatureBuilder&& UserOpSbpSignatureBuilder::Split(
    const std::vector<std::pair<std::string, int32_t>>& args, int64_t axis) {
  for (const auto& pair : args) {
    SplitImpl(sbp_sign_, GenRepeatedBn(pair.first, pair.second), axis);
  }
  return std::move(*this);
}

UserOpSbpSignatureBuilder&& UserOpSbpSignatureBuilder::Broadcast(const OpArg& op_arg) {
  BroadcastImpl(sbp_sign_, GenRepeatedBn(op_arg.name(), op_arg.index()));
  return std::move(*this);
}

UserOpSbpSignatureBuilder&& UserOpSbpSignatureBuilder::Broadcast(
    const std::vector<std::pair<std::string, int32_t>>& args) {
  for (const auto& pair : args) {
    BroadcastImpl(sbp_sign_, GenRepeatedBn(pair.first, pair.second));
  }
  return std::move(*this);
}

UserOpSbpSignatureBuilder&& UserOpSbpSignatureBuilder::PartialSum(const OpArg& op_arg) {
  PartialSumImpl(sbp_sign_, GenRepeatedBn(op_arg.name(), op_arg.index()));
  return std::move(*this);
}

UserOpSbpSignatureBuilder&& UserOpSbpSignatureBuilder::PartialSum(
    const std::vector<std::pair<std::string, int32_t>>& args) {
  for (const auto& pair : args) {
    PartialSumImpl(sbp_sign_, GenRepeatedBn(pair.first, pair.second));
  }
  return std::move(*this);
}

void SbpContext::AddSplitSbpSignList(int64_t num_axes) {
  for (int64_t i = 0; i < num_axes; ++i) {
    SbpSignature sbp_sign;
    for (const auto pair : inputs()) {
      (*sbp_sign.mutable_bn_in_op2sbp_parallel())[GenRepeatedBn(pair.first, pair.second)]
          .mutable_split_parallel()
          ->set_axis(i);
    }
    for (const auto pair : outputs()) {
      (*sbp_sign.mutable_bn_in_op2sbp_parallel())[GenRepeatedBn(pair.first, pair.second)]
          .mutable_split_parallel()
          ->set_axis(i);
    }
    *(sbp_sig_list_->mutable_sbp_signature()->Add()) = sbp_sign;
  }
}

Maybe<void> GetSbpFnUtil::MirrorSplitAtDim0(SbpContext* ctx) {
  SbpSignatureBuilder()
      .Split(ctx->inputs(), 0)
      .Split(ctx->outputs(), 0)
      .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

}  // namespace user_op

}  // namespace oneflow
