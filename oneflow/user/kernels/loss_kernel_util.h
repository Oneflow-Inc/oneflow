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
#ifndef ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {
namespace user_op {
namespace loss {

enum class ReductionType { kNone, kSum, kMean, kNotImplemented };

inline ReductionType GetReductionType(const std::string& reduction) {
  if (reduction == "mean") {
    return ReductionType::kMean;
  } else if (reduction == "none") {
    return ReductionType::kNone;
  } else if (reduction == "sum") {
    return ReductionType::kSum;
  }
  UNIMPLEMENTED();
  return ReductionType::kNotImplemented;
}

template<typename T>
void ApplyLossReduction(int64_t elem_cnt, const T* tmp_out, T* out,
                        const ReductionType reduction_type);

template<typename T>
void ApplyLossReduction(DeviceCtx* ctx, int64_t elem_cnt, const T* tmp_out, T* out,
                        const ReductionType reduction_type);

}  // namespace loss
}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_
