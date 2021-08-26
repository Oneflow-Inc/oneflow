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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/thread/thread_consistent_id.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/common/symbol.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<void> InitConsistentTransportTokenScope(const std::string& thread_tag,
                                              int64_t thread_consistent_id,
                                              Symbol<RankGroup> rank_group) {
  JUST(InitThisThreadUniqueConsistentId(thread_consistent_id, thread_tag));
  static thread_local const auto& init_rank_group_scope =
      JUST(RankGroupScope::MakeInitialRankGroupScope(rank_group));
  // no unused warning for `init_rank_group_scope`.
  (void)(init_rank_group_scope);
  return Maybe<void>::Ok();
}

Maybe<void> InitConsistentTransportTokenScope(const std::string& thread_tag,
                                              int64_t thread_consistent_id) {
  const auto& rank_group = JUST(RankGroup::DefaultRankGroup());
  JUST(InitConsistentTransportTokenScope(thread_tag, thread_consistent_id, rank_group));
  return Maybe<void>::Ok();
}

void ApiInitDefaultConsistentTransportTokenScope() {
  return InitConsistentTransportTokenScope("main", kThreadConsistentIdMain).GetOrThrow();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("InitDefaultConsistentTransportTokenScope", &ApiInitDefaultConsistentTransportTokenScope);
}

}  // namespace oneflow
