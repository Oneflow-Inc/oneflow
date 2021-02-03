// /*
// Copyright 2020 The OneFlow Authors. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// */
// #include "oneflow/user/kernels/elu_kernel.h"
// #include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

// namespace oneflow {

// template<>
// struct EluFunctor<half> {
//   OF_DEVICE_FUNC explicit EluFunctor(float alpha) : alpha(alpha), float_functor(EluFunctor<float>(alpha)) {}
//   OF_DEVICE_FUNC half operator()(half x) const { return __float2half(float_functor(__half2float(x))); }
//   const float alpha;
//   EluFunctor<float> float_functor;
// };

// template<>
// struct EluGradFunctor<half> {
//   OF_DEVICE_FUNC explicit EluGradFunctor(float alpha)
//       : alpha(alpha), float_functor(EluGradFunctor<float>(alpha)) {}
//   OF_DEVICE_FUNC half operator()(half x, half dy) const {
//     return __float2half(float_functor(__half2float(x), __half2float(dy)));
//   }
//   const float alpha;
//   EluGradFunctor<float> float_functor;
// };

// #define INSTANTIATE_ELU_GPU_FUNCTORS(dtype)                           \
//   INSTANTIATE_UNARY_XPU_FUNCTOR(DeviceType::kGPU, EluFunctor, dtype); \
//   INSTANTIATE_BINARY_XPU_FUNCTOR(DeviceType::kGPU, EluGradFunctor, dtype);

// INSTANTIATE_ELU_GPU_FUNCTORS(half);
// INSTANTIATE_ELU_GPU_FUNCTORS(double);
// INSTANTIATE_ELU_GPU_FUNCTORS(float);

// REGISTER_ELU_KERNEL(DeviceType::kGPU, half);
// REGISTER_ELU_KERNEL(DeviceType::kGPU, float);
// REGISTER_ELU_KERNEL(DeviceType::kGPU, double);

// }  // namespace oneflow
