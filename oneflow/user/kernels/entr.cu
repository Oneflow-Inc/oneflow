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
#include "oneflow/user/kernels/entr.h"
#include "oneflow/core/kernel/util/numerics.cuh"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"
namespace oneflow {
#ifdef WITH_CUDA
template<typename T>
struct EntrFunctor<DeviceType::kCUDA, T> {
  OF_DEVICE_FUNC T operator()(const T x) const {
    if (x > 0) {
      return -x * log(x);
    } else if (x == static_cast<T>(0)) {
      return static_cast<T>(0);
    } else {
      // -inf
      return -INFINITY;
    }
  }
};
template<typename T>
struct EntrGradFunctor<DeviceType::kCUDA, T> {
  OF_DEVICE_FUNC T operator()(const T x, const T dy) const {
    if (x > 0) {
      return (-log(x) - 1) * dy;
    } else if (x == static_cast<T>(0.0)) {
      // inf
      return INFINITY;
    } else {
      return detail::Nan<T>();
    }
  }
};
REGISTER_ENTR_KERNEL_DEVICE_TYPE(DeviceType::kCUDA, float);
REGISTER_ENTR_KERNEL_DEVICE_TYPE(DeviceType::kCUDA, double);
#endif  // WITH_CUDA
}  // namespace oneflow
