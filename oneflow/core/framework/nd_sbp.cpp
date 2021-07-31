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
    const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  static thread_local auto* sbp_list2nd_sbp_str =
      new HashMap<std::vector<Symbol<cfg::SbpParallel>>,
                  std::shared_ptr<std::vector<std::string>>>();
  auto iter = sbp_list2nd_sbp_str->find(sbp_list);
  if (iter == sbp_list2nd_sbp_str->end()) {
    std::shared_ptr<std::vector<std::string>> nd_sbp_str =
        std::make_shared<std::vector<std::string>>(sbp_list.size());
    for (int64_t i = 0; i < nd_sbp_str->size(); ++i) {
      nd_sbp_str->at(i) = SbpParallelToString(*sbp_list.at(i));
    }
    iter = sbp_list2nd_sbp_str->emplace(sbp_list, nd_sbp_str).first;
  }
  return iter->second;
}
}  // namespace

Maybe<Symbol<cfg::ParallelDistribution>> GetNdSbp(
    const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  return FindOrCreateNdSbp(sbp_list);
}

Maybe<std::vector<std::string>> GetNdSbpString(
    const std::vector<Symbol<cfg::SbpParallel>>& sbp_list) {
  return FindOrCreateNdSbpString(sbp_list);
}

}  // namespace oneflow
