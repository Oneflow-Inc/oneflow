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

Maybe<std::vector<std::string>> FindOrCreateNdSbpString(Symbol<cfg::NdSbp> nd_sbp) {
  static thread_local auto* nd_sbp2nd_sbp_str =
      new HashMap<Symbol<cfg::NdSbp>, std::shared_ptr<std::vector<std::string>>>();
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

Maybe<void> GetDualSbpParallel(const cfg::SbpParallel& sbp_parallel,
                               cfg::SbpParallel* dual_sbp_parallel) {
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

Maybe<Symbol<cfg::NdSbp>> GetDualNdSbp(Symbol<cfg::NdSbp> nd_sbp) {
  static thread_local HashMap<Symbol<cfg::NdSbp>, Symbol<cfg::NdSbp>> map;
  auto iter = map.find(nd_sbp);
  if (iter == map.end()) {
    cfg::NdSbp dual_nd_sbp;
    auto* mut_sbp_parallel = dual_nd_sbp.mutable_sbp_parallel();
    for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
      JUST(GetDualSbpParallel(sbp_parallel, mut_sbp_parallel->Add()));
    }
    iter = map.emplace(nd_sbp, SymbolOf(dual_nd_sbp)).first;
  }
  return iter->second;
}

Maybe<std::vector<std::string>> GetNdSbpStrList(
    const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  return FindOrCreateNdSbpString(JUST(GetNdSbp(sbp_list)));
}

Maybe<std::vector<std::string>> GetNdSbpStrList(Symbol<cfg::NdSbp> nd_sbp) {
  return FindOrCreateNdSbpString(nd_sbp);
}

Maybe<std::vector<std::string>> GetDualNdSbpStrList(Symbol<cfg::NdSbp> nd_sbp) {
  return GetNdSbpStrList(JUST(GetDualNdSbp(nd_sbp)));
}

namespace private_details {

Maybe<Symbol<cfg::NdSbp>> RawGetNdSbp(const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  CHECK_OR_RETURN(!sbp_list.empty());
  cfg::NdSbp nd_sbp;
  for (const auto& sbp : sbp_list) { *(nd_sbp.mutable_sbp_parallel()->Add()) = *sbp; }
  return SymbolOf(nd_sbp);
}

Maybe<std::vector<Symbol<cfg::SbpParallel>>> RawGetSbpList(Symbol<cfg::NdSbp> nd_sbp) {
  const auto& vec = std::make_shared<std::vector<Symbol<cfg::SbpParallel>>>();
  CHECK_OR_RETURN(!nd_sbp->sbp_parallel().empty());
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    vec->emplace_back(SymbolOf(sbp_parallel));
  }
  return vec;
}

}  // namespace private_details

const std::vector<Symbol<cfg::SbpParallel>>& GetNoneSbpList() {
  static thread_local std::vector<Symbol<cfg::SbpParallel>> none;
  return none;
}

std::string SbpToString(Symbol<cfg::SbpParallel> sbp_sym) { return SbpToString(*sbp_sym); }

std::string NdSbpToString(Symbol<cfg::NdSbp> nd_sbp_sym) { return NdSbpToString(*nd_sbp_sym); }

std::string SbpToString(const cfg::SbpParallel& sbp) {
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

std::string NdSbpToString(const cfg::NdSbp& nd_sbp) {
  std::ostringstream ss;
  ss << "(";
  for (size_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
    if (i > 0) { ss << ", "; }
    ss << SbpToString(nd_sbp.sbp_parallel(i));
  }
  ss << ")";
  return ss.str();
}

}  // namespace oneflow
