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
#include "oneflow/user/kernels/loss_kernel_util.h"
namespace oneflow {
namespace user_op {
namespace loss {

template<DeviceType device_type, typename T>
RETURN_VOID_IF_CPU(device_type)
ApplyLossReductionIfNeed(DeviceCtx* ctx, int64_t elem_cnt, const T* tmp_out, T* out,
                         const ReductionType reduction_type) {
  (void)ctx;
  if (reduction_type == ReductionType::kNone) { return; }
  if ((reduction_type != ReductionType::kMean) && (reduction_type != ReductionType::kSum)) {
    UNIMPLEMENTED();
    return;
  }
  *out = static_cast<T>(0);
  FOR_RANGE(int64_t, i, 0, elem_cnt) { *out += tmp_out[i]; }
  if (reduction_type == ReductionType::kMean) { *out /= elem_cnt; }
}

#define SPECIALIZE_APPLY_LOSS_REDUCTION(device_type, dtype)                              \
  template RETURN_VOID_IF_CPU(device_type) ApplyLossReductionIfNeed<device_type, dtype>( \
      DeviceCtx * ctx, int64_t elem_cnt, const dtype* tmp_out, dtype* out,               \
      const ReductionType reduction_type);

SPECIALIZE_APPLY_LOSS_REDUCTION(DeviceType::kCPU, float)
SPECIALIZE_APPLY_LOSS_REDUCTION(DeviceType::kCPU, double)

}  // namespace loss
}  // namespace user_op
}  // namespace oneflow
