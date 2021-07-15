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
#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SBP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SBP_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/sbp_infer_hint.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct ReduceSbpUtil final {
  static bool IsReduceAxisSplitted(const SbpInferHint& ibn_hint,
                                   const HashSet<int64_t>& reduced_axes);
  static std::function<bool(int32_t)> MakePredicatorIsReducedAxis(const HashSet<int32_t>& axes,
                                                                  int32_t num_axes);
  static std::function<bool(int32_t)> MakePredicatorIsReducedAxis(const PbRf<int32_t>& axes,
                                                                  int32_t num_axes);
  static void GetRegularAxes(int64_t num_axes, const std::vector<int32_t>& reduce_axes,
                             HashSet<int32_t>* axes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SBP_UTIL_H_
