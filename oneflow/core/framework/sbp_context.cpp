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

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Split(const OpArg& op_arg, int64_t axis) {
  SplitImpl(&sbp_sig_tmp_, GenRepeatedBn(op_arg.name(), op_arg.index()), axis);
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Split(const std::vector<OpArg>& op_args,
                                                            int64_t axis) {
  for (const auto& op_arg : op_args) { Split(op_arg, axis); }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Split(
    const std::vector<std::pair<std::string, int32_t>>& args, int64_t axis) {
  for (const auto& pair : args) {
    SplitImpl(&sbp_sig_tmp_, GenRepeatedBn(pair.first, pair.second), axis);
  }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Broadcast(const OpArg& op_arg) {
  BroadcastImpl(&sbp_sig_tmp_, GenRepeatedBn(op_arg.name(), op_arg.index()));
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Broadcast(const std::vector<OpArg>& op_args) {
  for (const auto& op_arg : op_args) { Broadcast(op_arg); }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::Broadcast(
    const std::vector<std::pair<std::string, int32_t>>& op_args) {
  for (const auto& pair : op_args) {
    BroadcastImpl(&sbp_sig_tmp_, GenRepeatedBn(pair.first, pair.second));
  }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::PartialSum(const OpArg& op_arg) {
  PartialSumImpl(&sbp_sig_tmp_, GenRepeatedBn(op_arg.name(), op_arg.index()));
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::PartialSum(
    const std::vector<OpArg>& op_args) {
  for (const auto& op_arg : op_args) { PartialSum(op_arg); }
  return *this;
}

UserOpSbpSignatureBuilder& UserOpSbpSignatureBuilder::PartialSum(
    const std::vector<std::pair<std::string, int32_t>>& op_args) {
  for (const auto& pair : op_args) {
    PartialSumImpl(&sbp_sig_tmp_, GenRepeatedBn(pair.first, pair.second));
  }
  return *this;
}

Maybe<void> GetSbpFnUtil::DefaultBroadcastToBroadcast(SbpContext* ctx) { return Maybe<void>::Ok(); }

}  // namespace user_op

}  // namespace oneflow
