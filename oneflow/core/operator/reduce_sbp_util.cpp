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
#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

bool ReduceSbpUtil::IsReduceAxisSplitted(const SbpInferHint& ibn_hint,
                                         const HashSet<int64_t>& reduced_axes) {
  if (ibn_hint.sbp_parallel().has_split_parallel() == false) { return false; }
  if (reduced_axes.empty()) { return true; }
  return reduced_axes.find(ibn_hint.sbp_parallel().split_parallel().axis()) != reduced_axes.end();
}

std::function<bool(int32_t)> ReduceSbpUtil::MakePredicatorIsReducedAxis(const PbRf<int32_t>& axes,
                                                                        int32_t num_axes) {
  HashSet<int32_t> axes_set = {axes.begin(), axes.end()};
  return MakePredicatorIsReducedAxis(axes_set, num_axes);
}

std::function<bool(int32_t)> ReduceSbpUtil::MakePredicatorIsReducedAxis(
    const HashSet<int32_t>& axes, int32_t num_axes) {
  auto axis_set = std::make_shared<HashSet<int32_t>>(axes);
  return [axis_set](int32_t axis) -> bool { return axis_set->find(axis) != axis_set->end(); };
}

void ReduceSbpUtil::GetRegularAxes(int64_t num_axes, const std::vector<int32_t>& reduce_axes,
                                   HashSet<int32_t>* axes) {
  for (auto axis : reduce_axes) { axes->insert(ShiftNegativeAxis(axis, num_axes)); }
}

}  // namespace oneflow
