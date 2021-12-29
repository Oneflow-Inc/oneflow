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
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"

namespace oneflow {
namespace {

static const double kUnsupportedBoxing = -1.0;

// check whether the sbp_parallel is legal
bool CheckSbpParallel(const cfg::SbpParallel& sbp_parallel) {
  // Which checking should we use?
  // return sbp_parallel.parallel_type_case() == SbpParallel::PARALLEL_TYPE_NOT_SET;
  return sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel()
         || sbp_parallel.has_partial_sum_parallel();
}

// check whether the nd_sbp is legal
bool CheckNdSbp(const cfg::NdSbp& nd_sbp) {
  if (nd_sbp.sbp_parallel_size() <= 0) { return false; }
  for (const auto& sbp : nd_sbp.sbp_parallel()) {
    if (!CheckSbpParallel(sbp)) { return false; }
  }
  return true;
}

double ComputCopyCostBetweenTwoDiffSbpParallel(const cfg::SbpParallel& producer_sbp_parallel,
                                               const cfg::SbpParallel& consumer_sbp_parallel,
                                               double logical_blob_size, double parallel_num,
                                               bool on_same_devices) {
  // Not supporting S->P for now.
  if (consumer_sbp_parallel.has_partial_sum_parallel()
      && producer_sbp_parallel.has_split_parallel()) {
    return kUnsupportedBoxing;
  }
  if (on_same_devices) {
    // B->S, B->P
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 0.0; }
    // has S
    if (consumer_sbp_parallel.has_split_parallel() || producer_sbp_parallel.has_split_parallel()) {
      if (consumer_sbp_parallel.has_split_parallel()
          && producer_sbp_parallel.has_split_parallel()) {
        // S(0)->S(1), S(1)->S(0), etc.
        return logical_blob_size * (parallel_num - 1) / parallel_num;
      } else {
        // P->S, S->B
        return logical_blob_size * (parallel_num - 1);
      }
    }
    // P->B (= p->S + S->B)
    return 2 * logical_blob_size * (parallel_num - 1);
  } else {
    // They have the same hierarchy at the transfer dimension.
    double overall_cost = logical_blob_size;
    // ? -> B
    if (consumer_sbp_parallel.has_broadcast_parallel()) {
      overall_cost += logical_blob_size * (parallel_num - 1);
    }
    // P -> ?
    if (producer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost += logical_blob_size * (parallel_num - 1);
    }
    // For B->P, B->S, S->S, overall_cost == logical_blob_size;
    return overall_cost;
  }
}

Maybe<double> ComputCopyCostBetweenTwoSbpParallel(const cfg::SbpParallel& producer_sbp_parallel,
                                                  const cfg::SbpParallel& consumer_sbp_parallel,
                                                  const BlobDesc& logical_blob_desc,
                                                  const ParallelDesc& producer_parallel_desc,
                                                  const ParallelDesc& consumer_parallel_desc,
                                                  bool is_same_sbp, bool allow_cpu2gpu) {
  if (!(CheckSbpParallel(producer_sbp_parallel) && CheckSbpParallel(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }

  // Not supporting S->P for now.
  if (consumer_sbp_parallel.has_partial_sum_parallel()
      && producer_sbp_parallel.has_split_parallel()) {
    return kUnsupportedBoxing;
  }
  if (producer_parallel_desc == consumer_parallel_desc) {
    // S->S, B->B, P->P
    if (producer_sbp_parallel == consumer_sbp_parallel) { return 0.0; }
    // Will directly modify output blob of source op. Requiring data having same sbp_parallel
    if (is_same_sbp) { return kUnsupportedBoxing; }
    // B->S, B->P
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 0.0; }
    double logical_blob_size =
        logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
    // has S
    if (consumer_sbp_parallel.has_split_parallel() || producer_sbp_parallel.has_split_parallel()) {
      if (consumer_sbp_parallel.has_split_parallel()
          && producer_sbp_parallel.has_split_parallel()) {
        // S(0)->S(1), S(1)->S(0), etc.
        return logical_blob_size * (producer_parallel_desc.parallel_num() - 1)
               / producer_parallel_desc.parallel_num();
      } else {
        // P->S, S->B
        return logical_blob_size * (producer_parallel_desc.parallel_num() - 1);
      }
    }
    // P->B (= p->S + S->B)
    return 2 * logical_blob_size * (producer_parallel_desc.parallel_num() - 1);
  } else {
    // Will directly modify output blob of source op. Requiring data having same sbp_parallel
    if (is_same_sbp
        && !(allow_cpu2gpu
             && producer_parallel_desc.EqualsIgnoringDeviceType(consumer_parallel_desc)
             && producer_sbp_parallel == consumer_sbp_parallel)) {
      return kUnsupportedBoxing;
    }
    double logical_blob_size =
        logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
    double overall_cost = logical_blob_size;
    // ? -> B
    if (consumer_sbp_parallel.has_broadcast_parallel()) {
      overall_cost += (consumer_parallel_desc.parallel_num() - 1) * logical_blob_size;
    }
    // P -> ?
    if (producer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost += (producer_parallel_desc.parallel_num() - 1) * logical_blob_size;
    }
    // For B->P, B->S, S->S, overall_cost == logical_blob_size;
    return overall_cost;
  }
}

Maybe<double> ComputCopyCostBetweenTwoNdSbp(const cfg::NdSbp& producer_nd_sbp,
                                            const cfg::NdSbp& consumer_nd_sbp,
                                            double logical_blob_size,
                                            const std::shared_ptr<Shape>& hierarchy,
                                            bool on_same_devices) {
  if (hierarchy->NumAxes() != 2) { return kUnsupportedBoxing; }
  const auto& producer_sbp_size = producer_nd_sbp.sbp_parallel_size();
  const auto& consumer_sbp_size = consumer_nd_sbp.sbp_parallel_size();
  // One of the SBP should have size 2
  CHECK_OR_RETURN((producer_sbp_size == 1 && consumer_sbp_size == 2)
                  || (producer_sbp_size == 2 && consumer_sbp_size == 1)
                  || (producer_sbp_size == 2 && consumer_sbp_size == 2))
      << "Not supporting such boxing type.";
  for (int32_t dim_same_sbp = 0; dim_same_sbp < 2; dim_same_sbp++) {
    // If the nd_sbp only have size 1, then make its dimension 0
    int32_t dim_producer = dim_same_sbp;
    if (producer_sbp_size == 1) { dim_producer = 0; }
    int32_t dim_consumer = dim_same_sbp;
    if (consumer_sbp_size == 1) { dim_consumer = 0; }
    // The SBP parallel are the same at dimension (dim_same_sbp)
    if (producer_nd_sbp.sbp_parallel(dim_producer) == consumer_nd_sbp.sbp_parallel(dim_consumer)) {
      if (!producer_nd_sbp.sbp_parallel(dim_producer).has_split_parallel()) {
        logical_blob_size *= hierarchy->At(dim_same_sbp);
      }
      // The SBP parallel are different at dimension (dim_diff_sbp)
      int32_t dim_diff_sbp = 1 - dim_same_sbp;
      // If the nd_sbp only have size 1, then make its dimension 0.
      // Since we have already do this before, we just maintain the value.
      // Otherwise, switch the dimension to dim_diff_sbp
      if (producer_sbp_size == 2) { dim_producer = dim_diff_sbp; }
      if (consumer_sbp_size == 2) { dim_consumer = dim_diff_sbp; }
      // Spliting at the same dimension needs special cares!
      // Not supported by nccl
      if (dim_diff_sbp == 0
          && producer_nd_sbp.sbp_parallel(dim_producer)
                 != consumer_nd_sbp.sbp_parallel(dim_consumer)) {
        return kUnsupportedBoxing;
      }
      return ComputCopyCostBetweenTwoDiffSbpParallel(
          producer_nd_sbp.sbp_parallel(dim_producer), consumer_nd_sbp.sbp_parallel(dim_consumer),
          logical_blob_size, hierarchy->At(dim_diff_sbp), on_same_devices);
    }
  }
  // (1, 2) || (2, 1):
  // Not support something like S0 -> (B, P)
  // (2, 2) :
  // if both dimensions are different, like (S0, S1) -> (S1, S0)
  // TODO: support it recently!
  // TODO: support it recently!
  // TODO: support it recently!
  return kUnsupportedBoxing;
}

}  // namespace

Maybe<Symbol<cfg::SbpParallel>> MakeSplitSbpParallel(int axis) {
  CHECK_LT_OR_RETURN(axis, kMaxSplitAxis);
  cfg::SbpParallel split_sbp_parallel;
  split_sbp_parallel.mutable_split_parallel()->set_axis(axis);
  return SymbolOf(split_sbp_parallel);
}

Maybe<Symbol<cfg::SbpParallel>> MakeBroadcastSbpParallel() {
  cfg::SbpParallel broadcast_sbp;
  broadcast_sbp.mutable_broadcast_parallel();
  return SymbolOf(broadcast_sbp);
}

Maybe<Symbol<cfg::SbpParallel>> MakePartialSumSbpParallel() {
  cfg::SbpParallel partial_sum_sbp;
  partial_sum_sbp.mutable_partial_sum_parallel();
  return SymbolOf(partial_sum_sbp);
}

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
    sorted_sbp_signatures->emplace_back(&sbp_signature);
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
  return SbpToString(sbp_parallel);
}

void SbpSignatureToNdSbpSignature(const cfg::SbpSignature& sbp_signature,
                                  cfg::NdSbpSignature* nd_sbp_signature) {
  for (const auto& pair : sbp_signature.bn_in_op2sbp_parallel()) {
    *((*nd_sbp_signature->mutable_bn_in_op2nd_sbp())[pair.first].add_sbp_parallel()) = pair.second;
  }
}

template<typename NdSbpSignatureT>
void NdSbpSignatureToSbpSignature(const NdSbpSignatureT& nd_sbp_signature,
                                  cfg::SbpSignature* sbp_signature) {
  for (const auto& pair : nd_sbp_signature.bn_in_op2nd_sbp()) {
    CHECK_EQ(pair.second.sbp_parallel_size(), 1);
    (*sbp_signature->mutable_bn_in_op2sbp_parallel())[pair.first] =
        cfg::SbpParallel(pair.second.sbp_parallel(0));
  }
}

template void NdSbpSignatureToSbpSignature(const NdSbpSignature& nd_sbp_signature,
                                           cfg::SbpSignature* sbp_signature);

template void NdSbpSignatureToSbpSignature(const cfg::NdSbpSignature& nd_sbp_signature,
                                           cfg::SbpSignature* sbp_signature);

void CheckSbpSignatureAndNdSbpEquals(const cfg::SbpSignature& sbp_sig,
                                     const cfg::NdSbpSignature& nd_sbp_sig) {
  CHECK_EQ(sbp_sig.bn_in_op2sbp_parallel_size(), nd_sbp_sig.bn_in_op2nd_sbp_size());
  for (const auto& pair : nd_sbp_sig.bn_in_op2nd_sbp()) {
    const auto& bn_in_op2sbp_parallel = sbp_sig.bn_in_op2sbp_parallel();
    const auto it = bn_in_op2sbp_parallel.find(pair.first);
    CHECK(it != bn_in_op2sbp_parallel.end());
    CHECK_EQ(pair.second.sbp_parallel_size(), 1);
    CHECK(pair.second.sbp_parallel(0) == it->second);
  }
}

Maybe<std::string> SbpSignatureListAsString(const cfg::SbpSignatureList& sbp_signatures,
                                            const PbRpf<std::string>& inputs,
                                            const PbRpf<std::string>& outputs) {
  std::ostringstream ss;
  if (sbp_signatures.sbp_signature_size() == 0) { return ss.str(); }

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

  JUST(WalkIO([](const std::string& bn) -> Maybe<std::string> { return bn; }));
  ss << ": ";

  ss << "[\n";
  for (const auto& sbp_signature : sbp_signatures.sbp_signature()) {
    ss << "\t";
    JUST(WalkIO([&](const std::string& bn) -> Maybe<std::string> {
      auto it = sbp_signature.bn_in_op2sbp_parallel().find(bn);
      if (it == sbp_signature.bn_in_op2sbp_parallel().end()) {
        return Error::RuntimeError()
               << "can't find " << bn << "in SbpSignature: " << sbp_signature.DebugString();
      }
      return SbpParallelToString(it->second);
    }));
    ss << ",\n";
  }
  ss << "]";
  return ss.str();
}

void ResizeNdSbpSignature(cfg::NdSbpSignature& nd_sbp_sig, int32_t size) {
  for (auto& pair : *nd_sbp_sig.mutable_bn_in_op2nd_sbp()) {
    if (pair.second.sbp_parallel_size() > size) { pair.second.clear_sbp_parallel(); }
    while (pair.second.sbp_parallel_size() < size) { pair.second.add_sbp_parallel(); }
  }
}

void SetNdSbpSignature(cfg::NdSbpSignature* nd_sbp_signature,
                       const cfg::SbpSignature& sbp_signature, int32_t sbp_axis) {
  for (const auto& pair : sbp_signature.bn_in_op2sbp_parallel()) {
    *((*nd_sbp_signature->mutable_bn_in_op2nd_sbp())[pair.first].mutable_sbp_parallel(sbp_axis)) =
        pair.second;
  }
}

void DfsGetNdSbpSignature(cfg::NdSbpSignature& nd_sbp_sig, int32_t depth, int32_t dims,
                          const cfg::SbpSignatureList& sbp_sig_list,
                          std::vector<cfg::NdSbpSignature>* nd_sbp_sig_list) {
  if (depth == dims) {
    nd_sbp_sig_list->push_back(nd_sbp_sig);
  } else {
    for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
      SetNdSbpSignature(&nd_sbp_sig, sbp_signature, depth);
      DfsGetNdSbpSignature(nd_sbp_sig, depth + 1, dims, sbp_sig_list, nd_sbp_sig_list);
    }
  }
}

Maybe<double> ComputCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                         const cfg::NdSbp& consumer_sbp_parallel,
                                         const BlobDesc& logical_blob_desc,
                                         const ParallelDesc& producer_parallel_desc,
                                         const ParallelDesc& consumer_parallel_desc,
                                         bool is_same_sbp, bool allow_cpu2gpu) {
  if (!(CheckNdSbp(producer_sbp_parallel) && CheckNdSbp(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }

  ParallelDesc reduced_in_parallel_desc = producer_parallel_desc;
  ParallelDesc reduced_out_parallel_desc = consumer_parallel_desc;
  cfg::NdSbp reduced_in_nd_sbp;
  cfg::NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(producer_parallel_desc, consumer_parallel_desc, producer_sbp_parallel,
                         consumer_sbp_parallel, &reduced_in_parallel_desc,
                         &reduced_out_parallel_desc, &reduced_in_nd_sbp, &reduced_out_nd_sbp);

  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();
  int32_t in_dim = in_hierarchy->NumAxes();
  int32_t out_dim = out_hierarchy->NumAxes();
  // TODO: Not supporting n-D sbp with n >= 3 now
  if (in_dim <= 0 || in_dim >= 3 || out_dim <= 0 || out_dim >= 3) { return kUnsupportedBoxing; }

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  if (same_nd_sbp && reduced_in_parallel_desc == reduced_out_parallel_desc) { return 0.0; }
  // Will directly modify output blob of source op. Requiring data having same sbp_parallel
  if (is_same_sbp
      && !(allow_cpu2gpu
           && reduced_in_parallel_desc.EqualsIgnoringDeviceType(reduced_out_parallel_desc)
           && same_nd_sbp)) {
    return kUnsupportedBoxing;
  }

  // We support different hierarchy for 1D sbp
  if (in_dim == 1 && out_dim == 1) {
    return ComputCopyCostBetweenTwoSbpParallel(
        reduced_in_nd_sbp.sbp_parallel(0), reduced_out_nd_sbp.sbp_parallel(0), logical_blob_desc,
        reduced_in_parallel_desc, reduced_out_parallel_desc, is_same_sbp, allow_cpu2gpu);
  }
  // Not supporting different hierarchy
  // TODO: Support it in the future
  if (in_hierarchy->elem_cnt() != out_hierarchy->elem_cnt()) { return kUnsupportedBoxing; }

  double logical_blob_size =
      logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
  bool on_same_devices =
      reduced_in_parallel_desc.EqualsIgnoringHierarchy(reduced_out_parallel_desc);

  if (in_dim == 2 && out_dim == 2) {
    // Not supporting different hierarchy
    // TODO: Support it in the future
    if (*in_hierarchy != *out_hierarchy) { return kUnsupportedBoxing; }
    return ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp, logical_blob_size,
                                         in_hierarchy, on_same_devices);
  }

  // (in_dim == 2 && out_dim == 1) || (in_dim == 1 && out_dim == 2)
  if (in_dim == 2 && out_dim == 1) {
    return ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp, logical_blob_size,
                                         in_hierarchy, on_same_devices);
  }

  if (in_dim == 1 && out_dim == 2) {
    return ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp, logical_blob_size,
                                         out_hierarchy, on_same_devices);
  }

  return Error::RuntimeError()
         << "Should not reach here. Something went wrong in ComputCopyCostBetweenNdSbp()";
}

}  // namespace oneflow
