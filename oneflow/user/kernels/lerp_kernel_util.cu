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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/lerp_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void LerpForwardGpu(const int n, const T* start, const T* weight, const T* end, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = start[i] + weight[i] * (end[i] - start[i]); }
}

template<typename T>
__global__ void LerpBackwardGpu(const int n, const T* start, const T* weight, const T* end,
                                const T* out_diff, T* start_diff, T* weight_diff, T* end_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T out_diff_i = out_diff[i];
    start_diff[i] = (static_cast<T>(1.0) - weight[i]) * out_diff_i;
    weight_diff[i] = (end[i] - start[i]) * out_diff_i;
    end_diff[i] = weight[i] * out_diff_i;
  }
}

}  // namespace

template<typename T>
struct LerpKernelUtil<DeviceType::kCUDA, T> {
  static void Forward(ep::Stream* stream, const int64_t n, const T* start, const T* weight,
                      const T* end, T* out) {
    RUN_CUDA_KERNEL((LerpForwardGpu<T>), stream, n, n, start, weight, end, out);
  }

  static void Backward(ep::Stream* stream, const int64_t n, const T* start, const T* weight,
                       const T* end, const T* out_diff, T* start_diff, T* weight_diff,
                       T* end_diff) {
    RUN_CUDA_KERNEL((LerpBackwardGpu<T>), stream, n, n, start, weight, end, out_diff,
                    start_diff, weight_diff, end_diff);
  }
};

#define INSTANTIATE_LERP_KERNEL_UTIL_CUDA(data_type, other) \
  template struct LerpKernelUtil<DeviceType::kCUDA, data_type>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LERP_KERNEL_UTIL_CUDA, LERP_DATA_TYPE_SEQ_CUDA)
#undef INSTANTIATE_LERP_KERNEL_UTIL_CUDA

}  // namespace oneflow
