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
#ifndef ONEFLOW_USER_KERNELS_SLICE_UTIL_H_
#define ONEFLOW_USER_KERNELS_SLICE_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

inline bool IsFullSlice(int64_t start, int64_t stop, int64_t step, int64_t size,
                        bool strict = true) {
  if (step != 1) { return false; }
  if (strict) {
    if (start != 0) { return false; }
    if (stop != std::numeric_limits<int64_t>::max()) { return false; }
  } else {
    if (start > 0) { return false; }
    if (stop < size) { return false; }
  }
  return true;
}

inline int64_t RegulateSliceStart(int64_t start, int64_t size) {
  // slice start must be in range [-size, size)
  // after changing to positive order it should be in range [0, size)
  start = std::min(std::max(start, -size), size - 1);
  return (start < 0) ? (start + size) : start;
}

inline int64_t RegulateSliceStop(int64_t stop, int64_t size) {
  // slice stop must be in range [-size-1, size]
  // after changing to positive order it should be in range [-1, size]
  stop = std::min(std::max(stop, -size - 1), size);
  return (stop < 0) ? (stop + size) : stop;
}

constexpr size_t kSliceMaxDims = 8;

struct SliceParams {
  int64_t ndim;
  int64_t dims[kSliceMaxDims];
  int64_t start[kSliceMaxDims];
  int64_t step[kSliceMaxDims];
  int64_t size[kSliceMaxDims];
};

SliceParams ConstructSliceParams(user_op::KernelComputeContext* ctx, const user_op::Tensor* entire,
                                 const user_op::Tensor* sliced);

template<DeviceType device_type, typename T>
struct SliceKernelUtil {
  static void Forward(DeviceCtx* ctx, const SliceParams& params, const T* entire, T* sliced);
  static void Backward(DeviceCtx* ctx, const SliceParams& params, const T* sliced, T* entire);
};

#define INSTANTIATE_SLICE_KERNEL_UTIL(device, dtype) template struct SliceKernelUtil<device, dtype>;

#define INSTANTIATE_SLICE_KERNEL_UTIL_WITH_DEVICE(device) \
  INSTANTIATE_SLICE_KERNEL_UTIL(device, float)            \
  INSTANTIATE_SLICE_KERNEL_UTIL(device, double)           \
  INSTANTIATE_SLICE_KERNEL_UTIL(device, int32_t)          \
  INSTANTIATE_SLICE_KERNEL_UTIL(device, int64_t)          \
  INSTANTIATE_SLICE_KERNEL_UTIL(device, int8_t)           \
  INSTANTIATE_SLICE_KERNEL_UTIL(device, uint8_t)

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_SLICE_UTIL_H_
