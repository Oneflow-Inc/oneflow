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
#include "oneflow/user/kernels/masked_fork_kernel.h"

namespace oneflow {
template<typename T>
struct MaskedForkFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const T* in, const int8_t* mask,
                  T* out_true, T* out_false) {
    ForkLoopFunctor<T>()(elem_cnt, in, mask, out_true, out_false);
  }
};
REGISTER_MASKED_FORK_KERNEL(DeviceType::kCPU, int8_t);
REGISTER_MASKED_FORK_KERNEL(DeviceType::kCPU, int32_t);
REGISTER_MASKED_FORK_KERNEL(DeviceType::kCPU, int64_t);
REGISTER_MASKED_FORK_KERNEL(DeviceType::kCPU, float);
REGISTER_MASKED_FORK_KERNEL(DeviceType::kCPU, double);
}  // namespace oneflow
