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

//  S -> S
//  P -> B
//  B -> P
cfg::SbpParallel GetDualSbpParallel(const cfg::SbpParallel& sbp_parallel) {
  cfg::SbpParallel ret(sbp_parallel);
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

bool IsSbpSignatureContaining(const cfg::SbpSignature& bigger, const cfg::SbpSignature& smaller) {
  auto& bn2sbp = bigger.bn_in_op2sbp_parallel();
  for (const auto& pair : smaller.bn_in_op2sbp_parallel()) {
    if (pair.second.parallel_type_case() == cfg::SbpParallel::PARALLEL_TYPE_NOT_SET) { continue; }
    CHECK(bn2sbp.find(pair.first) != bn2sbp.end());
    if (bn2sbp.at(pair.first) != pair.second) { return false; }
  }
  return true;
}

void FilterSbpSignatureList(const cfg::SbpSignatureList& sbp_sig_list,
                            const cfg::SbpSignature& sbp_sig_conf,
                            cfg::SbpSignatureList* filtered_sbp_sig_list) {
  for (const auto& sbp_sigature : sbp_sig_list.sbp_signature()) {
    if (IsSbpSignatureContaining(sbp_sigature, sbp_sig_conf)) {
      *filtered_sbp_sig_list->mutable_sbp_signature()->Add() = sbp_sigature;
    }
  }
}

double ComputCopyCostBetweenTwoSbpParallel(const SbpInferHint& producer_sbp_infer_hint,
                                           const cfg::SbpParallel& consumer_sbp_parallel) {
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
    const cfg::SbpSignature& sbp_signature) {
  double cost = 0;
  for (const auto& ibn : ibns) {
    const auto& consumer_sbp_parallel = sbp_signature.bn_in_op2sbp_parallel().find(ibn)->second;
    cost += ComputCopyCostBetweenTwoSbpParallel(*CHECK_JUST(SbpInferHint4Ibn(ibn)),
                                                consumer_sbp_parallel);
  }
  return cost;
}

std::function<double(const cfg::SbpSignature*)> MakeGetterIbnCopyCost4SbpSig(
    const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const cfg::SbpSignatureList& sbp_sig_list) {
  auto sbp_sig2ibn_copy_cast = std::make_shared<HashMap<const cfg::SbpSignature*, double>>();
  for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
    double cost = ComputeIbnCopyCost4SbpSig(ibns, SbpInferHint4Ibn, sbp_signature);
    CHECK(sbp_sig2ibn_copy_cast->emplace(&sbp_signature, cost).second);
  }
  return [sbp_sig2ibn_copy_cast](const cfg::SbpSignature* sbp_sig) -> double {
    return sbp_sig2ibn_copy_cast->at(sbp_sig);
  };
}

std::function<int32_t(const cfg::SbpSignature* sbp_sig)> MakeGetterOrderValue4SbpSig(
    const cfg::SbpSignatureList& sbp_sig_list,
    const std::function<int32_t(const cfg::SbpSignature&)>& CalcOrderValue4SbpSig) {
  auto sbp_sig2order_value = std::make_shared<HashMap<const cfg::SbpSignature*, int32_t>>();
  for (const cfg::SbpSignature& sbp_signature : sbp_sig_list.sbp_signature()) {
    sbp_sig2order_value->emplace(&sbp_signature, CalcOrderValue4SbpSig(sbp_signature));
  }
  return [sbp_sig2order_value](const cfg::SbpSignature* sbp_sig) {
    return sbp_sig2order_value->at(sbp_sig);
  };
}

void SortSbpSignatureListByCopyCost(
    const cfg::SbpSignatureList& sbp_sig_list, const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const std::function<int32_t(const cfg::SbpSignature&)>& CalcOrderValue4SbpSig,
    std::vector<const cfg::SbpSignature*>* sorted_sbp_signatures) {
  auto OrderValue4SbpSig = MakeGetterOrderValue4SbpSig(sbp_sig_list, CalcOrderValue4SbpSig);
  auto IbnCopyCost4SbpSig = MakeGetterIbnCopyCost4SbpSig(ibns, SbpInferHint4Ibn, sbp_sig_list);
  for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
    sorted_sbp_signatures->push_back(&sbp_signature);
  }
  std::sort(sorted_sbp_signatures->begin(), sorted_sbp_signatures->end(),
            [&](const cfg::SbpSignature* lhs, const cfg::SbpSignature* rhs) {
              if (OrderValue4SbpSig(lhs) < OrderValue4SbpSig(rhs)) { return true; }
              if (OrderValue4SbpSig(lhs) > OrderValue4SbpSig(rhs)) { return false; }
              return IbnCopyCost4SbpSig(lhs) < IbnCopyCost4SbpSig(rhs);
            });
}

bool IsValidSbpParallelString(const std::string& sbp_str) {
  cfg::SbpParallel sbp_parallel;
  return ParseSbpParallelFromString(sbp_str, &sbp_parallel);
}

bool ParseSbpParallelFromString(const std::string& sbp_str, cfg::SbpParallel* sbp_parallel) {
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

std::string SbpParallelToString(const cfg::SbpParallel& sbp_parallel) {
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
    const cfg::SbpSignature& sbp_signature,
    cfg::ParallelDistributionSignature* parallel_distribution_signature) {
  for (const auto& pair : sbp_signature.bn_in_op2sbp_parallel()) {
    *((*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())[pair.first]
          .add_sbp_parallel()) = pair.second;
  }
}

template<typename ParallelDistributionSignatureT>
void ParallelDistributionSignatureToSbpSignature(
    const ParallelDistributionSignatureT& parallel_distribution_signature,
    cfg::SbpSignature* sbp_signature) {
  for (const auto& pair : parallel_distribution_signature.bn_in_op2parallel_distribution()) {
    CHECK_EQ(pair.second.sbp_parallel_size(), 1);
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())[pair.first] = pair.second.sbp_parallel(0);
  }
}

template void ParallelDistributionSignatureToSbpSignature(
    const ParallelDistributionSignature& parallel_distribution_signature,
    cfg::SbpSignature* sbp_signature);

template void ParallelDistributionSignatureToSbpSignature(
    const cfg::ParallelDistributionSignature& parallel_distribution_signature,
    cfg::SbpSignature* sbp_signature);

void CheckSbpSignatureAndParallelDistributionEquals(
    const cfg::SbpSignature& sbp_sig,
    const cfg::ParallelDistributionSignature& parallel_distribution_sig) {
  CHECK_EQ(sbp_sig.bn_in_op2sbp_parallel_size(),
           parallel_distribution_sig.bn_in_op2parallel_distribution_size());
  for (const auto& pair : parallel_distribution_sig.bn_in_op2parallel_distribution()) {
    const auto& bn_in_op2sbp_parallel = sbp_sig.bn_in_op2sbp_parallel();
    const auto it = bn_in_op2sbp_parallel.find(pair.first);
    CHECK(it != bn_in_op2sbp_parallel.end());
    CHECK_EQ(pair.second.sbp_parallel_size(), 1);
    CHECK(pair.second.sbp_parallel(0) == it->second);
  }
}

}  // namespace oneflow
