/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
  CHECK_GE(num_axes_, 0);
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
