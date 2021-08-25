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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_UTIL_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

struct EagerBoxingInterpreterUtil {
  static bool IsBoxingS2S(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingS2B(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingS2P(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingP2S(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingP2B(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingB2S(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingB2P(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsAllBroadcastNdSbp(Symbol<cfg::NdSbp> nd_sbp);
  static bool IsAllPartialSumNdSbp(Symbol<cfg::NdSbp> nd_sbp);
  static bool IsAllSplitNdSbp(Symbol<cfg::NdSbp> nd_sbp, int64_t axis);
  static bool IsBroadcastSbp(const cfg::SbpParallel& sbp);
  static bool IsPartialSumSbp(const cfg::SbpParallel& sbp);
  static bool IsSplitSbp(const cfg::SbpParallel& sbp, int64_t axis);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_UTIL_H_
