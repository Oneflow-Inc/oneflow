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
#include "oneflow/user/kernels/narrow_util.h"

namespace oneflow {

template<typename T>
struct NarrowKernelUtil<DeviceType::kCPU, T> final {
  static void Forward(DeviceCtx* ctx, const T* in, const Shape& flat_in_shape, T* out,
                      const int64_t& start, const int64_t& length);
  static void Backward(DeviceCtx* ctx, const T* dy, const Shape& flat_in_shape, T* dx,
                       const int64_t& start, const int64_t& length);
};

template<typename T>
void NarrowKernelUtil<DeviceType::kCPU, T>::Forward(DeviceCtx* ctx, const T* in,
                                                    const Shape& flat_in_shape, T* out,
                                                    const int64_t& start, const int64_t& length) {
  const int64_t outer_dim_size = flat_in_shape.At(0);
  const int64_t narrow_dim_size = flat_in_shape.At(1);
  const int64_t inner_dim_size = flat_in_shape.At(2);
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, length) {
      T* to = out + outer_idx * length * inner_dim_size + i * inner_dim_size;
      const T* from =
          in + outer_idx * narrow_dim_size * inner_dim_size + (start + i) * inner_dim_size;
      std::copy(from, from + inner_dim_size, to);
    }
  }
}

template<typename T>
void NarrowKernelUtil<DeviceType::kCPU, T>::Backward(DeviceCtx* ctx, const T* dy,
                                                     const Shape& flat_in_shape, T* dx,
                                                     const int64_t& start, const int64_t& length) {
  const int64_t outer_dim_size = flat_in_shape.At(0);
  const int64_t narrow_dim_size = flat_in_shape.At(1);
  const int64_t inner_dim_size = flat_in_shape.At(2);
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, length) {
      const T* from = dy + outer_idx * length * inner_dim_size + i * inner_dim_size;
      T* to = dx + outer_idx * narrow_dim_size * inner_dim_size + (start + i) * inner_dim_size;
      std::copy(from, from + inner_dim_size, to);
    }
  }
}

INSTANTIATE_NARROW_KERNEL_UTIL_WITH_DEVICE(DeviceType::kCPU)

}  // namespace oneflow
