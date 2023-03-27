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
#ifndef ONEFLOW_USER_KERNELS_COMPLEX_KERNELS_UTIL_H_
#define ONEFLOW_USER_KERNELS_COMPLEX_KERNELS_UTIL_H_

namespace oneflow {
namespace user_op {

template<DeviceType device, typename dtype_x, typename dtype_out>
struct RealFunctor final {
  void operator()(ep::Stream* stream, const dtype_x* x, dtype_out* out);
};

#define INSTANTIATE_REAL_FUNCTOR(device, dtype_x, dtype_out)   \
  template struct RealFunctor<device, dtype_x, dtype_out>;

template<DeviceType device, typename dtype_x, typename dtype_out>
struct ImagFunctor final {
  void operator()(ep::Stream* stream, const dtype_x* x, dtype_out* out);
};

#define INSTANTIATE_IMAG_FUNCTOR(device, dtype_x, dtype_out)   \
  template struct ImagFunctor<device, dtype_x, dtype_out>;

template<DeviceType device, typename dtype>
struct ConjPhysicalFunctor final {
  void operator()(ep::Stream* stream, const dtype* x, dtype* out);
};

#define INSTANTIATE_CONJ_PHYSICAL_FUNCTOR(device, dtype)   \
  template struct ConjPhysicalFunctor<device, dtype>;

} // namespace user_op
} // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_COMPLEX_KERNELS_UTIL_H_
