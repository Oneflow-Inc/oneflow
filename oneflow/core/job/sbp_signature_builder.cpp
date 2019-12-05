#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

void SplitSbpSignatureListBuilder::CheckTemplate() {
  CHECK_GT(sbp_signature_template_.bn_in_op2sbp_parallel().size(), 0);
  const auto& first = sbp_signature_template_.bn_in_op2sbp_parallel().begin()->second;
  CHECK(first.has_split_parallel());
  for (const auto& pair : sbp_signature_template_.bn_in_op2sbp_parallel()) {
    CHECK(first == pair.second);
  }
}

SplitSbpSignatureListBuilder&& SplitSbpSignatureListBuilder::SetNumAxes(int64_t num_axes) {
  num_axes_ = num_axes;
  return std::move(*this);
}

void SplitSbpSignatureListBuilder::Build(SbpSignatureList* list) const {
  CHECK_GT(num_axes_, 0);
  SbpSignature sbp_sig_template(sbp_signature_template_);
  FOR_RANGE(int32_t, axis, 0, num_axes_) {
    for (auto& pair : *sbp_sig_template.mutable_bn_in_op2sbp_parallel()) {
      pair.second.mutable_split_parallel()->set_axis(axis);
    }
    *list->mutable_sbp_signature()->Add() = sbp_sig_template;
  }
}

SbpSignatureBuilder&& SbpSignatureBuilder::Split(const std::string& bn_in_op, int64_t axis) {
  (*sbp_signature_.mutable_bn_in_op2sbp_parallel())[bn_in_op].mutable_split_parallel()->set_axis(
      axis);
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Broadcast(const std::string& bn_in_op) {
  (*sbp_signature_.mutable_bn_in_op2sbp_parallel())[bn_in_op].mutable_broadcast_parallel();
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::PartialSum(const std::string& bn_in_op) {
  (*sbp_signature_.mutable_bn_in_op2sbp_parallel())[bn_in_op].mutable_partial_sum_parallel();
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Split(const std::string& arg_name, int32_t index,
                                                 int64_t axis) {
  Split(GenRepeatedBn(arg_name, index), axis);
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Broadcast(const std::string& arg_name, int32_t index) {
  Broadcast(GenRepeatedBn(arg_name, index));
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::PartialSum(const std::string& arg_name, int32_t index) {
  PartialSum(GenRepeatedBn(arg_name, index));
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Split(
    const std::vector<std::pair<std::string, int32_t>>& args, int64_t axis) {
  for (const auto& pair : args) { Split(GenRepeatedBn(pair.first, pair.second), axis); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Broadcast(
    const std::vector<std::pair<std::string, int32_t>>& args) {
  for (const auto& pair : args) { Broadcast(GenRepeatedBn(pair.first, pair.second)); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::PartialSum(
    const std::vector<std::pair<std::string, int32_t>>& args) {
  for (const auto& pair : args) { PartialSum(GenRepeatedBn(pair.first, pair.second)); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Split(const PbRpf<std::string>& bns, int64_t axis) {
  for (const auto& bn_in_op : bns) { Split(bn_in_op, axis); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Broadcast(const PbRpf<std::string>& bns) {
  for (const auto& bn_in_op : bns) { Broadcast(bn_in_op); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::PartialSum(const PbRpf<std::string>& bns) {
  for (const auto& bn_in_op : bns) { PartialSum(bn_in_op); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Split(const std::initializer_list<std::string>& bns,
                                                 int64_t axis) {
  for (const auto& bn_in_op : bns) { Split(bn_in_op, axis); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::Broadcast(
    const std::initializer_list<std::string>& bns) {
  for (const auto& bn_in_op : bns) { Broadcast(bn_in_op); }
  return std::move(*this);
}

SbpSignatureBuilder&& SbpSignatureBuilder::PartialSum(
    const std::initializer_list<std::string>& bns) {
  for (const auto& bn_in_op : bns) { PartialSum(bn_in_op); }
  return std::move(*this);
}

SplitSbpSignatureListBuilder SbpSignatureBuilder::MakeSplitSignatureListBuilder(
    int64_t num_axes) const {
  SbpSignature sbp_signature;
  Build(&sbp_signature);
  return SplitSbpSignatureListBuilder(sbp_signature).SetNumAxes(num_axes);
}

}  // namespace oneflow
