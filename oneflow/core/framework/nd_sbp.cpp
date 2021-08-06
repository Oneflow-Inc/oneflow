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
#include <mutex>
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

namespace {

Maybe<Symbol<cfg::ParallelDistribution>> FindOrCreateNdSbp(
    const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  static thread_local auto* sbp_list2nd_sbp =
      new HashMap<std::vector<Symbol<cfg::SbpParallel>>, Symbol<cfg::ParallelDistribution>>();
  auto iter = sbp_list2nd_sbp->find(sbp_list);
  if (iter == sbp_list2nd_sbp->end()) {
    cfg::ParallelDistribution parallel_distribution;
    for (Symbol<cfg::SbpParallel> sbp_symbol : sbp_list) {
      *(parallel_distribution.mutable_sbp_parallel()->Add()) = *sbp_symbol;
    }
    iter = sbp_list2nd_sbp->emplace(sbp_list, SymbolOf(parallel_distribution)).first;
  }
  return iter->second;
}

Maybe<std::vector<std::string>> FindOrCreateNdSbpString(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  static thread_local auto* parallel_distribution2nd_sbp_str =
      new HashMap<Symbol<cfg::ParallelDistribution>, std::shared_ptr<std::vector<std::string>>>();
  auto iter = parallel_distribution2nd_sbp_str->find(parallel_distribution);
  if (iter == parallel_distribution2nd_sbp_str->end()) {
    std::shared_ptr<std::vector<std::string>> nd_sbp_str =
        std::make_shared<std::vector<std::string>>(parallel_distribution->sbp_parallel_size());
    for (int64_t i = 0; i < nd_sbp_str->size(); ++i) {
      nd_sbp_str->at(i) = SbpParallelToString(parallel_distribution->sbp_parallel(i));
    }
    iter = parallel_distribution2nd_sbp_str->emplace(parallel_distribution, nd_sbp_str).first;
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

Maybe<Symbol<cfg::ParallelDistribution>> GetDualNdSbp(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  static thread_local HashMap<Symbol<cfg::ParallelDistribution>, Symbol<cfg::ParallelDistribution>>
      map;
  auto iter = map.find(parallel_distribution);
  if (iter == map.end()) {
    cfg::ParallelDistribution dual_parallel_distribution;
    auto* mut_sbp_parallel = dual_parallel_distribution.mutable_sbp_parallel();
    for (const auto& sbp_parallel : parallel_distribution->sbp_parallel()) {
      JUST(GetDualSbpParallel(sbp_parallel, mut_sbp_parallel->Add()));
    }
    iter = map.emplace(parallel_distribution, SymbolOf(dual_parallel_distribution)).first;
  }
  return iter->second;
}

Maybe<Symbol<cfg::ParallelDistribution>> GetNdSbp(
    const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  return FindOrCreateNdSbp(sbp_list);
}

Maybe<std::vector<std::string>> GetNdSbpStrList(
    const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  return FindOrCreateNdSbpString(JUST(GetNdSbp(sbp_list)));
}

Maybe<std::vector<std::string>> GetNdSbpStrList(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  return FindOrCreateNdSbpString(parallel_distribution);
}

Maybe<std::vector<std::string>> GetDualNdSbpStrList(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  return GetNdSbpStrList(JUST(GetDualNdSbp(parallel_distribution)));
}

}  // namespace oneflow
