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
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs) { return PbMd().Equals(lhs, rhs); }

bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs) { return !(lhs == rhs); }

bool operator==(const SbpSignature& lhs, const SbpSignature& rhs) {
  return PbMd().Equals(lhs, rhs);
}

bool operator!=(const SbpSignature& lhs, const SbpSignature& rhs) { return !(lhs == rhs); }

//  S -> S
//  P -> B
//  B -> P
SbpParallel GetDualSbpParallel(const SbpParallel& sbp_parallel) {
  SbpParallel ret(sbp_parallel);
  if (sbp_parallel.has_split_parallel()) {
    //  do nothing
  } else if (sbp_parallel.has_broadcast_parallel()) {
    ret.mutable_partial_sum_parallel();
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    ret.mutable_broadcast_parallel();
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

bool operator==(const ParallelDistribution& lhs, const ParallelDistribution& rhs) {
  return PbMd().Equals(lhs, rhs);
}

bool operator!=(const ParallelDistribution& lhs, const ParallelDistribution& rhs) {
  return !(lhs == rhs);
}

bool operator==(const ParallelDistributionSignature& lhs,
                const ParallelDistributionSignature& rhs) {
  return PbMd().Equals(lhs, rhs);
}

bool operator!=(const ParallelDistributionSignature& lhs,
                const ParallelDistributionSignature& rhs) {
  return !(lhs == rhs);
}

bool IsSbpSignatureContaining(const SbpSignature& bigger, const SbpSignature& smaller) {
  auto& bn2sbp = bigger.bn_in_op2sbp_parallel();
  for (const auto& pair : smaller.bn_in_op2sbp_parallel()) {
    if (pair.second.parallel_type_case() == SbpParallel::PARALLEL_TYPE_NOT_SET) { continue; }
    CHECK(bn2sbp.find(pair.first) != bn2sbp.end());
    if (bn2sbp.at(pair.first) != pair.second) { return false; }
  }
  return true;
}

void FilterSbpSignatureList(const SbpSignatureList& sbp_sig_list, const SbpSignature& sbp_sig_conf,
                            SbpSignatureList* filtered_sbp_sig_list) {
  for (const auto& sbp_sigature : sbp_sig_list.sbp_signature()) {
    if (IsSbpSignatureContaining(sbp_sigature, sbp_sig_conf)) {
      *filtered_sbp_sig_list->mutable_sbp_signature()->Add() = sbp_sigature;
    }
  }
}

double ComputCopyCostBetweenTwoSbpParallel(const SbpInferHint& producer_sbp_infer_hint,
                                           const SbpParallel& consumer_sbp_parallel) {
  if (producer_sbp_infer_hint.sbp_parallel() == consumer_sbp_parallel) { return 0.0; }
  if (consumer_sbp_parallel.has_partial_sum_parallel()) { return GetMaxVal<int64_t>(); }
  if (producer_sbp_infer_hint.sbp_parallel().has_broadcast_parallel()) {
    return GetMaxVal<int32_t>();
  }
  const auto& logical_blob_desc = producer_sbp_infer_hint.logical_blob_desc();
  return logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
}

double ComputeIbnCopyCost4SbpSig(
    const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const SbpSignature& sbp_signature) {
  double cost = 0;
  for (const auto& ibn : ibns) {
    const auto& consumer_sbp_parallel = sbp_signature.bn_in_op2sbp_parallel().find(ibn)->second;
    cost += ComputCopyCostBetweenTwoSbpParallel(*CHECK_JUST(SbpInferHint4Ibn(ibn)),
                                                consumer_sbp_parallel);
  }
  return cost;
}

std::function<double(const SbpSignature*)> MakeGetterIbnCopyCost4SbpSig(
    const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const SbpSignatureList& sbp_sig_list) {
  auto sbp_sig2ibn_copy_cast = std::make_shared<HashMap<const SbpSignature*, double>>();
  for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
    double cost = ComputeIbnCopyCost4SbpSig(ibns, SbpInferHint4Ibn, sbp_signature);
    CHECK(sbp_sig2ibn_copy_cast->emplace(&sbp_signature, cost).second);
  }
  return [sbp_sig2ibn_copy_cast](const SbpSignature* sbp_sig) -> double {
    return sbp_sig2ibn_copy_cast->at(sbp_sig);
  };
}

std::function<int32_t(const SbpSignature* sbp_sig)> MakeGetterOrderValue4SbpSig(
    const SbpSignatureList& sbp_sig_list,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig) {
  auto sbp_sig2order_value = std::make_shared<HashMap<const SbpSignature*, int32_t>>();
  for (const SbpSignature& sbp_signature : sbp_sig_list.sbp_signature()) {
    sbp_sig2order_value->emplace(&sbp_signature, CalcOrderValue4SbpSig(sbp_signature));
  }
  return [sbp_sig2order_value](const SbpSignature* sbp_sig) {
    return sbp_sig2order_value->at(sbp_sig);
  };
}

void SortSbpSignatureListByCopyCost(
    const SbpSignatureList& sbp_sig_list, const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::vector<const SbpSignature*>* sorted_sbp_signatures) {
  auto OrderValue4SbpSig = MakeGetterOrderValue4SbpSig(sbp_sig_list, CalcOrderValue4SbpSig);
  auto IbnCopyCost4SbpSig = MakeGetterIbnCopyCost4SbpSig(ibns, SbpInferHint4Ibn, sbp_sig_list);
  for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
    sorted_sbp_signatures->push_back(&sbp_signature);
  }
  std::sort(sorted_sbp_signatures->begin(), sorted_sbp_signatures->end(),
            [&](const SbpSignature* lhs, const SbpSignature* rhs) {
              if (OrderValue4SbpSig(lhs) < OrderValue4SbpSig(rhs)) { return true; }
              if (OrderValue4SbpSig(lhs) > OrderValue4SbpSig(rhs)) { return false; }
              return IbnCopyCost4SbpSig(lhs) < IbnCopyCost4SbpSig(rhs);
            });
}

bool IsValidSbpParallelString(const std::string& sbp_str) {
  SbpParallel sbp_parallel;
  return ParseSbpParallelFromString(sbp_str, &sbp_parallel);
}

bool ParseSbpParallelFromString(const std::string& sbp_str, SbpParallel* sbp_parallel) {
  bool success = false;
  if (sbp_str.length() >= 1) {
    if (sbp_str == "B") {
      sbp_parallel->mutable_broadcast_parallel();
      success = true;
    } else if (sbp_str == "P") {
      sbp_parallel->mutable_partial_sum_parallel();
      success = true;
    } else if (sbp_str[0] == 'S') {
      if (sbp_str.length() >= 4 && sbp_str[1] == '(' && sbp_str[sbp_str.length() - 1] == ')') {
        int split_axis = 0;
        if (sbp_str.length() == 4) {
          split_axis = sbp_str[2] - '0';
          if (split_axis >= 0 && split_axis <= 9) { success = true; }
        } else {
          std::string split_axis_str = sbp_str.substr(2, sbp_str.length() - 3);
          if (std::all_of(split_axis_str.cbegin(), split_axis_str.cend(),
                          [](char ch) { return std::isdigit(ch); })) {
            size_t pos = 0;
            split_axis = std::stoi(split_axis_str, &pos);
            if (pos == split_axis_str.length()) { success = true; }
          }
        }
        if (success) { sbp_parallel->mutable_split_parallel()->set_axis(split_axis); }
      }
    }
  }
  return success;
}

std::string SbpParallelToString(const SbpParallel& sbp_parallel) {
  std::string sbp_str = "";
  if (sbp_parallel.has_broadcast_parallel()) {
    sbp_str = "B";
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    sbp_str = "P";
  } else if (sbp_parallel.has_split_parallel()) {
    sbp_str = "S(" + std::to_string(sbp_parallel.split_parallel().axis()) + ")";
  } else {
    UNIMPLEMENTED();
  }
  return sbp_str;
}

void SbpSignatureToParallelDistributionSignature(
    const SbpSignature& sbp_signature,
    ParallelDistributionSignature* parallel_distribution_signature) {
  for (const auto& pair : sbp_signature.bn_in_op2sbp_parallel()) {
    *((*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())[pair.first]
          .add_sbp_parallel()) = pair.second;
  }
}

void ParallelDistributionSignatureToSbpSignature(
    const ParallelDistributionSignature& parallel_distribution_signature,
    SbpSignature* sbp_signature) {
  for (const auto& pair : parallel_distribution_signature.bn_in_op2parallel_distribution()) {
    CHECK_EQ(pair.second.sbp_parallel_size(), 1);
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())[pair.first] = pair.second.sbp_parallel(0);
  }
}

}  // namespace oneflow
