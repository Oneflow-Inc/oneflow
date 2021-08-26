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
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"

namespace oneflow {

bool EagerBoxingInterpreterUtil::IsBoxingS2S(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_split_parallel() && dst.has_split_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingS2B(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_split_parallel() && dst.has_broadcast_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingS2P(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_split_parallel() && dst.has_partial_sum_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingP2S(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_partial_sum_parallel() && dst.has_split_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingP2B(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_partial_sum_parallel() && dst.has_broadcast_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingB2S(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_broadcast_parallel() && dst.has_split_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingB2P(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_broadcast_parallel() && dst.has_partial_sum_parallel();
}

bool EagerBoxingInterpreterUtil::IsAllBroadcastNdSbp(Symbol<cfg::NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_broadcast_parallel()) { return false; }
  }
  return true;
}

bool EagerBoxingInterpreterUtil::IsAllPartialSumNdSbp(Symbol<cfg::NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_partial_sum_parallel()) { return false; }
  }
  return true;
}

bool EagerBoxingInterpreterUtil::IsAllSplitNdSbp(Symbol<cfg::NdSbp> nd_sbp, int64_t axis) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!(sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == axis)) {
      return false;
    }
  }
  return true;
}

bool EagerBoxingInterpreterUtil::IsBroadcastSbp(const cfg::SbpParallel& sbp) {
  return sbp.has_broadcast_parallel();
}

bool EagerBoxingInterpreterUtil::IsPartialSumSbp(const cfg::SbpParallel& sbp) {
  return sbp.has_partial_sum_parallel();
}

bool EagerBoxingInterpreterUtil::IsSplitSbp(const cfg::SbpParallel& sbp, int64_t axis) {
  return (sbp.has_split_parallel() && sbp.split_parallel().axis() == axis);
}

}  // namespace oneflow
