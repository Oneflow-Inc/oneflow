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
#ifndef ONEFLOW_UNFOLD_TENSOR_KERNEL_UTILS_H_
#define ONEFLOW_UNFOLD_TENSOR_KERNEL_UTILS_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {
OF_DEVICE_FUNC int32_t Offset(int32_t in_offset, const int32_t* out_stride,
                              const int32_t* out_shape, const int32_t n) {
  int32_t remaining = 0;
  int32_t out_offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (int32_t dim = n; dim >= 0; --dim) {
    remaining = in_offset % out_shape[dim];
    out_offset += remaining * out_stride[dim];
    in_offset = in_offset / out_shape[dim];
  }
  return out_offset;
}

}  // namespace oneflow

#endif  // ONEFLOW_UNFOLD_TENSOR_KERNEL_UTILS_H_
