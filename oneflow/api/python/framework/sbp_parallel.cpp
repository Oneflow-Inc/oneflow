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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace py = pybind11;

namespace oneflow {

static const int64_t kAxisNumMax = 6;

namespace {

std::vector<std::shared_ptr<cfg::SbpParallel>> MakeSplitSbpParallelList(int axis_num_max) {
  std::vector<std::shared_ptr<cfg::SbpParallel>> ret(axis_num_max);
  for (int i = 0; i < axis_num_max; ++i) {
    std::shared_ptr<cfg::SbpParallel> split_sbp_parallel = std::make_shared<cfg::SbpParallel>();
    split_sbp_parallel->mutable_split_parallel()->set_axis(i);
    ret[i] = (split_sbp_parallel);
  }
  return ret;
}

Maybe<cfg::SbpParallel> GetSplitSbpParallel(int axis) {
  CHECK_LT_OR_RETURN(axis, kAxisNumMax);
  static std::vector<std::shared_ptr<cfg::SbpParallel>> split_sbp_list =
      MakeSplitSbpParallelList(kAxisNumMax);
  return split_sbp_list.at(axis);
}

Maybe<cfg::SbpParallel> GetBroadcastSbpParallel() {
  std::shared_ptr<cfg::SbpParallel> broadcast_sbp = std::make_shared<cfg::SbpParallel>();
  broadcast_sbp->mutable_broadcast_parallel();
  return broadcast_sbp;
}

Maybe<cfg::SbpParallel> GetPartialSumSbpParallel() {
  std::shared_ptr<cfg::SbpParallel> partial_sum_sbp = std::make_shared<cfg::SbpParallel>();
  partial_sum_sbp->mutable_partial_sum_parallel();
  return partial_sum_sbp;
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("sbp", m) {
  m.def("split", [](int axis) { return GetSplitSbpParallel(axis).GetPtrOrThrow(); });
  m.def("broadcast", []() { return GetBroadcastSbpParallel().GetPtrOrThrow(); });
  m.def("partial_sum", []() { return GetPartialSumSbpParallel().GetPtrOrThrow(); });
}

}  // namespace oneflow
