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
#ifndef ONEFLOW_ROLL_KERNEL_UTILS_H_
#define ONEFLOW_ROLL_KERNEL_UTILS_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

const int32_t NDIMS = 16;

struct SHIFTS {
  int32_t val[NDIMS];
};

struct SHAPE {
  int32_t val[NDIMS];
};

struct STRIDE {
  int32_t val[NDIMS];
};

OF_DEVICE_FUNC int32_t getShiftedOffset(const int32_t offset, const int32_t* shifts,
                                        const int32_t* shape, const int32_t* stride,
                                        const int32_t n) {
  int32_t remaining = offset;
  int32_t out_offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (int32_t i = 0; i < n; ++i) {
    const int32_t idx = remaining / stride[i];
    // NOTE(Liang Depeng):
    // Compute the shifted index of each axis.
    // In C++ the sign of the result of `%` operation is the same of the dividend,
    // but we want to the circle index when `idx - shifts[i]` is negative.
    // So we just add the divisor `shape[i]` to the result and do the `%` operation one more time.
    const int32_t shifted_idx = (((idx - shifts[i]) % shape[i]) + shape[i]) % shape[i];
    out_offset += shifted_idx * stride[i];
    remaining = remaining - idx * stride[i];
  }
  return out_offset;
}

static void initStride(int32_t* stride, const int32_t* dim_vec, const int32_t dims) {
  stride[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; --i) { stride[i] = dim_vec[i + 1] * stride[i + 1]; }
}

static void computeParams(const ShapeView& in_shape, const std::vector<int32_t>& shifts,
                          const std::vector<int32_t>& dims, int32_t* new_shifts, int32_t* new_shape,
                          int32_t* new_num_axes) {
  if (dims[0] == -1) {
    // NOTE(Liang Depeng):
    // If user did not set the dims parameter,
    // the input tensor will be flattened before rolling,
    // which means we can think of the input tensor as an 1 dimensional array.
    new_shifts[0] = shifts[0];
    *new_num_axes = 1;
    new_shape[0] = in_shape.elem_cnt();
  } else {
    std::map<int32_t, int32_t> dim_to_shift;
    for (int i = 0; i < shifts.size(); ++i) { dim_to_shift.emplace(dims[i], shifts[i]); }
    // NOTE(Liang Depeng):
    // Compute the shift parameter for each axis.
    // For those axis which user did not specified shift value, will be set to 0
    for (int i = 0; i < in_shape.NumAxes(); ++i) {
      if (dim_to_shift.count(i) > 0) {
        new_shifts[i] = dim_to_shift.at(i);
      } else {
        new_shifts[i] = 0;
      }
      new_shape[i] = in_shape.At(i);
    }
    *new_num_axes = in_shape.NumAxes();
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_ROLL_KERNEL_UTILS_H_
