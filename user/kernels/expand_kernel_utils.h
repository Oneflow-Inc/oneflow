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
#ifndef ONEFLOW_EXPAND_KERNEL_UTILS_H_
#define ONEFLOW_EXPAND_KERNEL_UTILS_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

OF_DEVICE_FUNC int32_t OffsetToNdIndexToOffset(const int32_t offset, const int32_t* in_stride,
                                               const int32_t* out_stride, const int32_t n) {
  int32_t remaining = offset;
  int32_t out_offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (int32_t i = 0; i < n; ++i) {
    const int32_t idx = remaining / in_stride[i];
    out_offset += idx * out_stride[i];
    remaining = remaining - idx * in_stride[i];
  }
  return out_offset;
}

static void InitStride(int32_t* stride, const int32_t* dim_vec, const int32_t dims) {
  stride[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; --i) { stride[i] = dim_vec[i + 1] * stride[i + 1]; }
}

Maybe<void> getOutShapeAndStride(const std::vector<int32_t>& in_shape,
                                 const std::vector<int32_t>& expand_shape,
                                 std::vector<int32_t>& out_shape, std::vector<int32_t>& stride);

Maybe<void> getOutShapeAndStrideForFp(const std::vector<int32_t>& in_shape,
                                      const std::vector<int32_t>& logical_expand_shape,
                                      std::vector<int32_t>& out_shape,
                                      std::vector<int32_t>& stride);

Maybe<void> getOutShapeAndStrideForBp(const std::vector<int32_t>& logical_out_shape,
                                      const std::vector<int32_t>& logical_expand_shape,
                                      const std::vector<int32_t>& in_shape,
                                      std::vector<int32_t>& out_shape,
                                      std::vector<int32_t>& stride);

}  // namespace oneflow

#endif  // ONEFLOW_EXPAND_KERNEL_UTILS_H_
