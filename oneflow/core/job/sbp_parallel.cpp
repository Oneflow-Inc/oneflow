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
#include "oneflow/core/framework/nd_sbp.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs) {
  if (lhs.parallel_type_case() != rhs.parallel_type_case()) { return false; }
  if (lhs.has_split_parallel()) {
    return lhs.split_parallel().axis() == rhs.split_parallel().axis();
  } else if (lhs.has_broadcast_parallel()) {
    return true;
  } else if (lhs.has_partial_sum_parallel()) {
    return true;
  } else {
    UNIMPLEMENTED();
  }
}

bool operator==(const NdSbp& lhs, const NdSbp& rhs) {
  if (lhs.sbp_parallel_size() != rhs.sbp_parallel_size()) { return false; }
  for (int i = 0; i < lhs.sbp_parallel_size(); ++i) {
    if (lhs.sbp_parallel(i) != rhs.sbp_parallel(i)) { return false; }
  }
  return true;
  ;
}

bool operator==(const SbpSignature& lhs, const SbpSignature& rhs) {
  if (lhs.bn_in_op2sbp_parallel_size() != rhs.bn_in_op2sbp_parallel_size()) { return false; }
  const auto& lhs_map = lhs.bn_in_op2sbp_parallel();
  const auto& rhs_map = rhs.bn_in_op2sbp_parallel();
  for (const auto& lhs_pair : lhs_map) {
    const auto& rhs_iter = rhs_map.find(lhs_pair.first);
    if (rhs_iter == rhs_map.end()) { return false; }
    if (lhs_pair.second != rhs_iter->second) { return false; }
  }
  return true;
}

bool operator==(const NdSbpSignature& lhs, const NdSbpSignature& rhs) {
  if (lhs.bn_in_op2nd_sbp_size() != rhs.bn_in_op2nd_sbp_size()) { return false; }
  const auto& lhs_map = lhs.bn_in_op2nd_sbp();
  const auto& rhs_map = rhs.bn_in_op2nd_sbp();
  for (const auto& lhs_pair : lhs_map) {
    const auto& rhs_iter = rhs_map.find(lhs_pair.first);
    if (rhs_iter == rhs_map.end()) { return false; }
    if (lhs_pair.second != rhs_iter->second) { return false; }
  }
  return true;
}

Maybe<Symbol<SbpParallel>> MakeSplitSbpParallel(int axis) {
  CHECK_LT_OR_RETURN(axis, kMaxSplitAxis);
  SbpParallel split_sbp_parallel;
  split_sbp_parallel.mutable_split_parallel()->set_axis(axis);
  return SymbolOf(split_sbp_parallel);
}

Maybe<Symbol<SbpParallel>> MakeBroadcastSbpParallel() {
  SbpParallel broadcast_sbp;
  broadcast_sbp.mutable_broadcast_parallel();
  return SymbolOf(broadcast_sbp);
}

Maybe<Symbol<SbpParallel>> MakePartialSumSbpParallel() {
  SbpParallel partial_sum_sbp;
  partial_sum_sbp.mutable_partial_sum_parallel();
  return SymbolOf(partial_sum_sbp);
}

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

bool IsSbpSignatureContaining(const SbpSignature& bigger, const SbpSignature& smaller) {
  auto& bn2sbp = bigger.bn_in_op2sbp_parallel();
  for (const auto& pair : smaller.bn_in_op2sbp_parallel()) {
    if (pair.second.parallel_type_case() == SbpParallel::PARALLEL_TYPE_NOT_SET) { continue; }
    CHECK(bn2sbp.find(pair.first) != bn2sbp.end()) << pair.first;
    if (bn2sbp.at(pair.first) != pair.second) { return false; }
  }
  return true;
}

void FilterSbpSignatureList(const SbpSignatureList& sbp_sig_list, const SbpSignature& sbp_sig_conf,
                            SbpSignatureList* filtered_sbp_sig_list) {
  for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
    if (IsSbpSignatureContaining(sbp_signature, sbp_sig_conf)) {
      *filtered_sbp_sig_list->mutable_sbp_signature()->Add() = sbp_signature;
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
    sorted_sbp_signatures->emplace_back(&sbp_signature);
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
  return SbpToString(sbp_parallel);
}

bool ParseNdSbpFromStringList(const std::vector<std::string>& sbp_str_list, NdSbp* nd_sbp) {
  for (const auto& sbp_str : sbp_str_list) {
    if (!ParseSbpParallelFromString(sbp_str, nd_sbp->add_sbp_parallel())) { return false; }
  }
  return true;
}

std::vector<std::string> NdSbpToStringList(const NdSbp& nd_sbp) {
  std::vector<std::string> sbp_str_list(nd_sbp.sbp_parallel_size());
  for (size_t i = 0; i < sbp_str_list.size(); ++i) {
    sbp_str_list[i] = SbpToString(nd_sbp.sbp_parallel(i));
  }
  return sbp_str_list;
}

void SbpSignatureToNdSbpSignature(const SbpSignature& sbp_signature,
                                  NdSbpSignature* nd_sbp_signature) {
  for (const auto& pair : sbp_signature.bn_in_op2sbp_parallel()) {
    *((*nd_sbp_signature->mutable_bn_in_op2nd_sbp())[pair.first].add_sbp_parallel()) = pair.second;
  }
}

void NdSbpSignatureToSbpSignature(const NdSbpSignature& nd_sbp_signature,
                                  SbpSignature* sbp_signature) {
  for (const auto& pair : nd_sbp_signature.bn_in_op2nd_sbp()) {
    CHECK_EQ(pair.second.sbp_parallel_size(), 1);
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())[pair.first] = pair.second.sbp_parallel(0);
  }
}

void CheckSbpSignatureAndNdSbpEquals(const SbpSignature& sbp_sig,
                                     const NdSbpSignature& nd_sbp_sig) {
  CHECK_EQ(sbp_sig.bn_in_op2sbp_parallel_size(), nd_sbp_sig.bn_in_op2nd_sbp_size());
  for (const auto& pair : nd_sbp_sig.bn_in_op2nd_sbp()) {
    const auto& bn_in_op2sbp_parallel = sbp_sig.bn_in_op2sbp_parallel();
    const auto it = bn_in_op2sbp_parallel.find(pair.first);
    CHECK(it != bn_in_op2sbp_parallel.end());
    CHECK_EQ(pair.second.sbp_parallel_size(), 1);
    CHECK(pair.second.sbp_parallel(0) == it->second);
  }
}

bool NdSbpAllSameSplitParallel(const NdSbp& nd_sbp) {
  CHECK_GT(nd_sbp.sbp_parallel_size(), 0);
  const SbpParallel& first_sbp = nd_sbp.sbp_parallel(0);
  if (!first_sbp.has_split_parallel()) { return false; }
  FOR_RANGE(int64_t, i, 1, nd_sbp.sbp_parallel_size()) {
    if (nd_sbp.sbp_parallel(i) != first_sbp) { return false; }
  }
  return true;
}

Maybe<std::string> NdSbpSignatureToString(const NdSbpSignature& nd_sbp_signature,
                                          const std::vector<std::string>& inputs,
                                          const std::vector<std::string>& outputs) {
  std::ostringstream ss;

  auto AppendBnNdSbpString = [&](const std::string& bn) -> Maybe<void> {
    auto iter = nd_sbp_signature.bn_in_op2nd_sbp().find(bn);
    if (iter == nd_sbp_signature.bn_in_op2nd_sbp().end()) {
      return Error::RuntimeError()
             << "can't find " << bn << " in NdSbpSignature: " << nd_sbp_signature.DebugString();
    }
    ss << " " << NdSbpToString(iter->second);
    return Maybe<void>::Ok();
  };

  int bn_index = 0;
  for (const auto& ibn : inputs) {
    if (bn_index > 0) { ss << ", "; }
    ss << ibn;
    JUST(AppendBnNdSbpString(ibn));
    bn_index++;
  }

  ss << " -> ";
  bn_index = 0;
  for (const auto& obn : outputs) {
    if (bn_index > 0) { ss << ", "; }
    ss << obn;
    JUST(AppendBnNdSbpString(obn));
    bn_index++;
  }

  return ss.str();
}

Maybe<std::string> NdSbpSignatureToString(const NdSbpSignature& nd_sbp_signature,
                                          const PbRpf<std::string>& inputs,
                                          const PbRpf<std::string>& outputs) {
  return NdSbpSignatureToString(nd_sbp_signature,
                                std::vector<std::string>{inputs.begin(), inputs.end()},
                                std::vector<std::string>{outputs.begin(), outputs.end()});
}

Maybe<std::string> NdSbpSignatureListToString(const std::vector<NdSbpSignature>& nd_sbp_sig_list,
                                              const std::vector<std::string>& inputs,
                                              const std::vector<std::string>& outputs) {
  std::ostringstream ss;
  if (nd_sbp_sig_list.empty()) { return ss.str(); }

  auto WalkIO =
      [&](const std::function<Maybe<std::string>(const std::string&)>& bn_handler) -> Maybe<void> {
    ss << "(";
    for (size_t i = 0; i < inputs.size(); ++i) {
      ss << *JUST(bn_handler(inputs[i]));
      if (i != inputs.size() - 1) { ss << ", "; }
    }
    ss << ") -> (";
    for (size_t i = 0; i < outputs.size(); ++i) {
      ss << *JUST(bn_handler(outputs[i]));
      if (i != outputs.size() - 1) { ss << ", "; }
    }
    ss << ")";
    return Maybe<void>::Ok();
  };

  ss << "\n";
  JUST(WalkIO([](const std::string& bn) -> Maybe<std::string> { return bn; }));
  ss << ": ";

  ss << "[\n";
  for (const auto& nd_sbp_sig : nd_sbp_sig_list) {
    ss << "\t";
    JUST(WalkIO([&](const std::string& bn) -> Maybe<std::string> {
      auto it = nd_sbp_sig.bn_in_op2nd_sbp().find(bn);
      if (it == nd_sbp_sig.bn_in_op2nd_sbp().end()) {
        return Error::RuntimeError()
               << "can't find " << bn << " in NdSbpSignature: " << nd_sbp_sig.DebugString();
      }
      return NdSbpToString(it->second);
    }));
    ss << ",\n";
  }
  ss << "]";
  return ss.str();
}

Maybe<std::string> NdSbpSignatureListToString(const std::vector<NdSbpSignature>& nd_sbp_sig_list,
                                              const PbRpf<std::string>& inputs,
                                              const PbRpf<std::string>& outputs) {
  return NdSbpSignatureListToString(nd_sbp_sig_list,
                                    std::vector<std::string>{inputs.begin(), inputs.end()},
                                    std::vector<std::string>{outputs.begin(), outputs.end()});
}

}  // namespace oneflow
