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
#include "oneflow/user/kernels/activation_kernels.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

namespace oneflow {

template<>
struct EluFunctor<half> {
  OF_DEVICE_FUNC explicit EluFunctor(float alpha)
      : alpha(alpha), float_functor(EluFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  const float alpha;
  EluFunctor<float> float_functor;
};

template<>
struct EluGradFunctor<half> {
  OF_DEVICE_FUNC explicit EluGradFunctor(float alpha)
      : alpha(alpha), float_functor(EluGradFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  const float alpha;
  EluGradFunctor<float> float_functor;
};

template<>
struct HardswishFunctor<half> {
  HardswishFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
};

template<>
struct HardswishGradFunctor<half> {
  HardswishGradFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
};

template<>
struct MishFunctor<half> {
  OF_DEVICE_FUNC explicit MishFunctor() : float_functor(MishFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  MishFunctor<float> float_functor;
};

template<>
struct MishGradFunctor<half> {
  OF_DEVICE_FUNC explicit MishGradFunctor() : float_functor(MishGradFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  MishGradFunctor<float> float_functor;
};

template<>
struct SiluFunctor<half> {
  OF_DEVICE_FUNC explicit SiluFunctor() : float_functor(SiluFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  SiluFunctor<float> float_functor;
};

template<>
struct SiluGradFunctor<half> {
  OF_DEVICE_FUNC explicit SiluGradFunctor() : float_functor(SiluGradFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  SiluGradFunctor<float> float_functor;
};

template<>
struct SeluFunctor<half> {
  OF_DEVICE_FUNC explicit SeluFunctor() : float_functor(SeluFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  SeluFunctor<float> float_functor;
};

template<>
struct SeluGradFunctor<half> {
  OF_DEVICE_FUNC explicit SeluGradFunctor() : float_functor(SeluGradFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  SeluGradFunctor<float> float_functor;
};

template<>
struct SoftSignFunctor<half> {
  OF_DEVICE_FUNC explicit SoftSignFunctor() : float_functor(SoftSignFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  SoftSignFunctor<float> float_functor;
};

template<>
struct SoftSignGradFunctor<half> {
  OF_DEVICE_FUNC explicit SoftSignGradFunctor() : float_functor(SoftSignGradFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  SoftSignGradFunctor<float> float_functor;
};

#define REGISTER_ACTIVATION_GPU_KERNEL(dtype)           \
  REGISTER_ELU_KERNEL(DeviceType::kGPU, dtype);         \
  REGISTER_HARDSWISH_KERNEL(DeviceType::kGPU, dtype);   \
  REGISTER_HARDSIGMOID_KERNEL(DeviceType::kGPU, dtype); \
  REGISTER_HARDTANH_KERNEL(DeviceType::kGPU, dtype);    \
  REGISTER_MISH_KERNEL(DeviceType::kGPU, dtype);        \
  REGISTER_SILU_KERNEL(DeviceType::kGPU, dtype);        \
  REGISTER_SELU_KERNEL(DeviceType::kGPU, dtype);        \
  REGISTER_SOFTSIGN_KERNEL(DeviceType::kGPU, dtype);

REGISTER_ACTIVATION_GPU_KERNEL(half);
REGISTER_ACTIVATION_GPU_KERNEL(float);
REGISTER_ACTIVATION_GPU_KERNEL(double);

}  // namespace oneflow
