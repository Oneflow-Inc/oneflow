/*
Copyright 2023 The OneFlow Authors. All rights reserved.

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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/complex_kernels_util.h"
#include <complex>

namespace oneflow {

namespace user_op {

template<typename dtype_x, typename dtype_out>
struct RealFunctor<DeviceType::kCPU, dtype_x, dtype_out> final {
  void operator()(ep::Stream* stream, const dtype_x* x, dtype_out* out) {
    // TODO(lml): finish this function.
  }
};

INSTANTIATE_REAL_FUNCTOR(DeviceType::kCPU, std::complex<float>, float)
INSTANTIATE_REAL_FUNCTOR(DeviceType::kCPU, std::complex<double>, double)

template<typename dtype_x, typename dtype_out>
struct ImagFunctor<DeviceType::kCPU, dtype_x, dtype_out> final {
  void operator()(ep::Stream* stream, const dtype_x* x, dtype_out* out) {
    // TODO(lml): finish this function.
  }
};

INSTANTIATE_IMAG_FUNCTOR(DeviceType::kCPU, std::complex<float>, float)
INSTANTIATE_IMAG_FUNCTOR(DeviceType::kCPU, std::complex<double>, double)

template<typename dtype>
struct ConjPhysicalFunctor<DeviceType::kCPU, dtype> final {
  void operator()(ep::Stream* stream, const dtype* x, dtype* out) {
    // TODO(lml): finish this function.
  }
};

INSTANTIATE_CONJ_PHYSICAL_FUNCTOR(DeviceType::kCPU, std::complex<float>)
INSTANTIATE_CONJ_PHYSICAL_FUNCTOR(DeviceType::kCPU, std::complex<double>)

}  // namespace user_op
}  // namespace oneflow
