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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

namespace {

Maybe<std::vector<std::string>> FindOrCreateNdSbpString(Symbol<NdSbp> nd_sbp) {
  static thread_local auto* nd_sbp2nd_sbp_str =
      new HashMap<Symbol<NdSbp>, std::shared_ptr<std::vector<std::string>>>();
  auto iter = nd_sbp2nd_sbp_str->find(nd_sbp);
  if (iter == nd_sbp2nd_sbp_str->end()) {
    std::shared_ptr<std::vector<std::string>> nd_sbp_str =
        std::make_shared<std::vector<std::string>>(nd_sbp->sbp_parallel_size());
    for (int64_t i = 0; i < nd_sbp_str->size(); ++i) {
      nd_sbp_str->at(i) = SbpParallelToString(nd_sbp->sbp_parallel(i));
    }
    iter = nd_sbp2nd_sbp_str->emplace(nd_sbp, nd_sbp_str).first;
  }
  return iter->second;
}

Maybe<void> GetDualSbpParallel(const SbpParallel& sbp_parallel, SbpParallel* dual_sbp_parallel) {
  if (sbp_parallel.has_split_parallel()) {
    *dual_sbp_parallel = sbp_parallel;
  } else if (sbp_parallel.has_broadcast_parallel()) {
    dual_sbp_parallel->mutable_partial_sum_parallel();
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    dual_sbp_parallel->mutable_broadcast_parallel();
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<Symbol<NdSbp>> GetDualNdSbp(Symbol<NdSbp> nd_sbp) {
  static thread_local HashMap<Symbol<NdSbp>, Symbol<NdSbp>> map;
  auto iter = map.find(nd_sbp);
  if (iter == map.end()) {
    NdSbp dual_nd_sbp;
    auto* mut_sbp_parallel = dual_nd_sbp.mutable_sbp_parallel();
    for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
      JUST(GetDualSbpParallel(sbp_parallel, mut_sbp_parallel->Add()));
    }
    iter = map.emplace(nd_sbp, SymbolOf(dual_nd_sbp)).first;
  }
  return iter->second;
}

Maybe<std::vector<std::string>> GetNdSbpStrList(const std::vector<Symbol<SbpParallel>>& sbp_list) {
  return FindOrCreateNdSbpString(JUST(GetNdSbp(sbp_list)));
}

Maybe<std::vector<std::string>> GetNdSbpStrList(Symbol<NdSbp> nd_sbp) {
  return FindOrCreateNdSbpString(nd_sbp);
}

Maybe<std::vector<std::string>> GetDualNdSbpStrList(Symbol<NdSbp> nd_sbp) {
  return GetNdSbpStrList(JUST(GetDualNdSbp(nd_sbp)));
}

namespace private_details {

Maybe<Symbol<NdSbp>> RawGetNdSbp(const std::vector<Symbol<SbpParallel>>& sbp_list) {
  CHECK_OR_RETURN(!sbp_list.empty())
      << Error::InvalidValueError() << "sbp_list should be non-empty";
  NdSbp nd_sbp;
  for (const auto& sbp : sbp_list) { *(nd_sbp.mutable_sbp_parallel()->Add()) = *sbp; }
  return SymbolOf(nd_sbp);
}

Maybe<std::vector<Symbol<SbpParallel>>> RawGetSbpList(Symbol<NdSbp> nd_sbp) {
  const auto& vec = std::make_shared<std::vector<Symbol<SbpParallel>>>();
  CHECK_OR_RETURN(!nd_sbp->sbp_parallel().empty())
      << Error::InvalidValueError() << "sbp_parallel should be non-empty";
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    vec->emplace_back(SymbolOf(sbp_parallel));
  }
  return vec;
}

bool RawContainSplitSbp(Symbol<NdSbp> nd_sbp) {
  for (int32_t i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
    if (nd_sbp->sbp_parallel(i).has_split_parallel()) { return true; }
  }
  return false;
}

Maybe<std::vector<Symbol<SbpParallel>>> RawNdSbpReplacePartialByBroadcast(
    const std::vector<Symbol<SbpParallel>>& sbp_list) {
  auto result = std::make_shared<std::vector<Symbol<SbpParallel>>>(sbp_list.size());
  for (int i = 0; i < sbp_list.size(); ++i) {
    const auto& sbp = sbp_list[i];
    if (sbp->has_partial_sum_parallel()) {
      (*result)[i] = JUST(MakeBroadcastSbpParallel());
    } else {
      (*result)[i] = sbp;
    }
  }
  return result;
}

}  // namespace private_details

const std::vector<Symbol<SbpParallel>>& GetNoneSbpList() {
  static thread_local std::vector<Symbol<SbpParallel>> none;
  return none;
}

std::string SbpToString(Symbol<SbpParallel> sbp_sym) { return SbpToString(*sbp_sym); }

std::string NdSbpToString(Symbol<NdSbp> nd_sbp_sym) { return NdSbpToString(*nd_sbp_sym); }

std::string SbpToString(const SbpParallel& sbp) {
  std::ostringstream ss;
  if (sbp.has_broadcast_parallel()) {
    ss << "B";
  } else if (sbp.has_partial_sum_parallel()) {
    ss << "P";
  } else if (sbp.has_split_parallel()) {
    ss << "S(" << std::to_string(sbp.split_parallel().axis()) << ")";
  } else {
    UNIMPLEMENTED();
  }
  return ss.str();
}

std::string NdSbpToString(const NdSbp& nd_sbp) {
  std::ostringstream ss;
  ss << "(";
  for (size_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
    if (i > 0) { ss << ", "; }
    ss << SbpToString(nd_sbp.sbp_parallel(i));
  }
  ss << ")";
  return ss.str();
}

Maybe<Symbol<NdSbp>> SetSbpAtAxis(Symbol<NdSbp> nd_sbp, Symbol<SbpParallel> sbp, int axis) {
  return SetSbpAtAxis(*nd_sbp, *sbp, axis);
}

Maybe<Symbol<NdSbp>> SetSbpAtAxis(const NdSbp& nd_sbp, const SbpParallel& sbp, int axis) {
  CHECK_LT_OR_RETURN(axis, nd_sbp.sbp_parallel_size())
      << Error::RuntimeError() << "Expected axis to be less than the size of sbp list ("
      << nd_sbp.sbp_parallel_size() << "), but got " << axis;
  NdSbp out_nd_sbp = nd_sbp;
  *out_nd_sbp.mutable_sbp_parallel(axis) = sbp;
  return SymbolOf(out_nd_sbp);
}

Maybe<Symbol<NdSbp>> SbpToNdSbp(Symbol<SbpParallel> sbp) { return SbpToNdSbp(*sbp); }

Maybe<Symbol<NdSbp>> SbpToNdSbp(const SbpParallel& sbp) {
  NdSbp out_nd_sbp;
  *out_nd_sbp.add_sbp_parallel() = sbp;
  return SymbolOf(out_nd_sbp);
}

// If an nd sbp can be converted to a 1d sbp.
bool Is1dSbp(const NdSbp& nd_sbp) {
  if (nd_sbp.sbp_parallel_size() == 0) { return false; }
  // Equivalent to
  // return std::all_of(nd_sbp.sbp_parallel().begin() + 1, nd_sbp.sbp_parallel().end(),
  //                    [&](const auto& sbp) { return sbp == nd_sbp.sbp_parallel(0); });
  for (int32_t i = 1; i < nd_sbp.sbp_parallel_size(); i++) {
    if (nd_sbp.sbp_parallel(0) != nd_sbp.sbp_parallel(i)) { return false; }
  }
  return true;
}

}  // namespace oneflow
