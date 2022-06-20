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
struct LeakyReluGradFunctor<half> {
  OF_DEVICE_FUNC explicit LeakyReluGradFunctor(float alpha) : alpha(alpha) {}
  __device__ half operator()(half x, half dy) const {
    half zero = __float2half(0);
    return (x > zero) ? dy : __float2half(alpha) * dy;
  }
  const float alpha;
};

template<>
struct SoftplusGradFunctor<half> {
  OF_DEVICE_FUNC explicit SoftplusGradFunctor(float beta, float threshold)
      : beta(beta),
        threshold(threshold),
        float_functor(SoftplusGradFunctor<float>(beta, threshold)) {}
  __device__ half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  const float beta;
  const float threshold;
  SoftplusGradFunctor<float> float_functor;
};

template<>
struct CeluGradFunctor<half> {
  OF_DEVICE_FUNC explicit CeluGradFunctor(float alpha)
      : alpha(alpha), float_functor(CeluGradFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  const float alpha;
  CeluGradFunctor<float> float_functor;
};

template<>
struct HardswishGradFunctor<half> {
  HardswishGradFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
};

template<>
struct HardShrinkGradFunctor<half> {
  OF_DEVICE_FUNC explicit HardShrinkGradFunctor(float lambd)
      : lambd(lambd), float_functor(HardShrinkGradFunctor<float>(lambd)) {}
  OF_DEVICE_FUNC half operator()(half y, half dy) const {
    return __float2half(float_functor(__half2float(y), __half2float(dy)));
  }

  const float lambd;
  HardShrinkGradFunctor<float> float_functor;
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
struct SiluGradFunctor<half> {
  OF_DEVICE_FUNC explicit SiluGradFunctor() : float_functor(SiluGradFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  SiluGradFunctor<float> float_functor;
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
struct SoftSignGradFunctor<half> {
  OF_DEVICE_FUNC explicit SoftSignGradFunctor() : float_functor(SoftSignGradFunctor<float>()) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  SoftSignGradFunctor<float> float_functor;
};

template<>
struct ThresholdGradFunctor<half> {
  OF_DEVICE_FUNC explicit ThresholdGradFunctor(float threshold)
      : threshold(threshold), float_functor(ThresholdGradFunctor<float>(threshold)) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }

  const float threshold;
  ThresholdGradFunctor<float> float_functor;
};

template<>
struct ReluGradFunctor<half> {
  OF_DEVICE_FUNC explicit ReluGradFunctor() {}
  __device__ half operator()(half y, half dy) const {
    half zero = __float2half(0.0);
    if (__hgt(y, zero)) {
      return dy;
    } else {
      return zero;
    }
  }
};

template<>
struct SoftShrinkGradFunctor<half> {
  OF_DEVICE_FUNC explicit SoftShrinkGradFunctor(float alpha)
      : alpha(alpha), float_functor(SoftShrinkGradFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half y, half dy) const {
    return __float2half(float_functor(__half2float(y), __half2float(dy)));
  }

  const float alpha;
  SoftShrinkGradFunctor<float> float_functor;
};

#define REGISTER_ACTIVATION_CUDA_KERNEL(dtype)                    \
  REGISTER_ELU_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);         \
  REGISTER_CELU_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_HARDSWISH_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);   \
  REGISTER_HARDSIGMOID_BACKWARD_KERNEL(DeviceType::kCUDA, dtype); \
  REGISTER_HARDSHRINK_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);  \
  REGISTER_HARDTANH_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);    \
  REGISTER_MISH_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_SILU_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_SELU_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_SOFTSHRINK_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);  \
  REGISTER_SOFTSIGN_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);    \
  REGISTER_LEAKYRELU_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);   \
  REGISTER_THRESHOLD_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);   \
  REGISTER_SOFTPLUS_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);    \
  REGISTER_RELU_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);

namespace {

REGISTER_ACTIVATION_CUDA_KERNEL(half);
REGISTER_ACTIVATION_CUDA_KERNEL(float);
REGISTER_ACTIVATION_CUDA_KERNEL(double);

}  // namespace

}  // namespace oneflow
