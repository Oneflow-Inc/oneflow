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
#ifndef ONEFLOW_USER_KERNELS_CLIP_BY_VALUE_KERNEL_H_
#define ONEFLOW_USER_KERNELS_CLIP_BY_VALUE_KERNEL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

template<typename T>
OF_DEVICE_FUNC T DeviceMin(T a, T b) {
#if defined(__CUDA_ARCH__)
  return a < b ? a : b;
#else
  return std::min(a, b);
#endif
}

template<typename T>
OF_DEVICE_FUNC T DeviceMax(T a, T b) {
#if defined(__CUDA_ARCH__)
  return a > b ? a : b;
#else
  return std::max(a, b);
#endif
}

template<typename T>
struct ClipByMinFunctor {
  ClipByMinFunctor(T min) : min_value(min) {}
  OF_DEVICE_FUNC T operator()(T value) { return DeviceMax(value, min_value); }
  T min_value;
};

template<typename T>
struct ClipByMaxFunctor {
  ClipByMaxFunctor(T max) : max_value(max) {}
  OF_DEVICE_FUNC T operator()(T value) { return DeviceMin(value, max_value); }
  T max_value;
};

template<typename T>
struct ClipByMinMaxFunctor {
  ClipByMinMaxFunctor(T min, T max) : min_value(min), max_value(max) {}
  OF_DEVICE_FUNC T operator()(T value) { return DeviceMin(DeviceMax(value, min_value), max_value); }
  T min_value;
  T max_value;
};

template<typename T>
struct ClipByMinGradFunctor {
  ClipByMinGradFunctor(T min) : min_value(min) {}
  OF_DEVICE_FUNC T operator()(T value, T grad) {
    return value < min_value ? static_cast<T>(0) : grad;
  }
  T min_value;
};

template<typename T>
struct ClipByMaxGradFunctor {
  ClipByMaxGradFunctor(T max) : max_value(max) {}
  OF_DEVICE_FUNC T operator()(T value, T grad) {
    return value > max_value ? static_cast<T>(0) : grad;
  }
  T max_value;
};

template<typename T>
struct ClipByMinMaxGradFunctor {
  ClipByMinMaxGradFunctor(T min, T max) : min_value(min), max_value(max) {}
  OF_DEVICE_FUNC T operator()(T value, T grad) {
    return (value < min_value || value > max_value) ? static_cast<T>(0) : grad;
  }
  T min_value;
  T max_value;
};

template<DeviceType device_type, typename T>
struct ClipKernelUtil {
  template<typename F>
  static void Forward(ep::Stream* stream, F clip_func, const int64_t n, const T* x, T* y);
  template<typename F>
  static void Backward(ep::Stream* stream, F clip_func, const int64_t n, const T* x, const T* dy,
                       T* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CLIP_BY_VALUE_KERNEL_H_
