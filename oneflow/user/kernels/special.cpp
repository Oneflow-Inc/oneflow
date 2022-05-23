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
#include "oneflow/user/kernels/special.h"
#include "oneflow/core/common/device_type.pb.h"
namespace oneflow {
template<typename T>
struct EntrFunctor<DeviceType::kCPU, T> {
  OF_DEVICE_FUNC T operator()(const T x) const {
    if (x > static_cast<T>(0)) {
      return -x * std::log(x);
    } else if (x == static_cast<T>(0)) {
      return static_cast<T>(0);
    } else {
      return -std::numeric_limits<T>::infinity();
    }
  }
};

template<typename T>
struct EntrGradFunctor<DeviceType::kCPU, T> {
  OF_DEVICE_FUNC T operator()(const T x, const T dy) const {
    if (x > static_cast<T>(0)) {
      return (-std::log(x) - 1) * dy;
    } else if (x == static_cast<T>(0.0)) {
      return std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::quiet_NaN();
    }
  }
};

template<typename T>
struct ErfFunctor<DeviceType::kCPU, T> {
  OF_DEVICE_FUNC T operator()(const T x) const { return std::erf(x); }
};

template<typename T>
struct ErfGradFunctor<DeviceType::kCPU, T> {
  OF_DEVICE_FUNC T operator()(const T x, const T dy) const {
    return dy * 2.0 * std::exp(-x * x) / std::sqrt(x);
  }
};

#define REGISTER_SPECIAL_OPS_CPU_KERNEL(kernel_name, func_prefix)                             \
  REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, DeviceType::kCPU, float); \
  REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, DeviceType::kCPU, double);
OF_PP_FOR_EACH_TUPLE(REGISTER_SPECIAL_OPS_CPU_KERNEL, SPECIAL_UNARY_OPS)
#undef REGISTER_SPECIAL_OPS_CPU_KERNEL
}  // namespace oneflow
