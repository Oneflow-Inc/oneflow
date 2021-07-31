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

bool EagerBoxingInterpreterUtil::IsPlacementSymmetrical(Symbol<ParallelDesc> src,
                                                        Symbol<ParallelDesc> dst) {
  return src == dst;
}

bool EagerBoxingInterpreterUtil::IsDeviceTypeGPU(Symbol<ParallelDesc> parallel_desc) {
  return parallel_desc->device_type() == DeviceType::kGPU;
}

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

bool EagerBoxingInterpreterUtil::IsBoxingP2P(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_partial_sum_parallel() && dst.has_partial_sum_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingB2B(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_broadcast_parallel() && dst.has_broadcast_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingB2S(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_broadcast_parallel() && dst.has_split_parallel();
}

bool EagerBoxingInterpreterUtil::IsBoxingB2P(const cfg::SbpParallel& src,
                                             const cfg::SbpParallel& dst) {
  return src.has_broadcast_parallel() && dst.has_partial_sum_parallel();
}
}  // namespace oneflow
