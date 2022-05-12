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
struct LeakyReluFunctor<half> {
  OF_DEVICE_FUNC explicit LeakyReluFunctor(float alpha) : alpha(alpha) {}
  __device__ half operator()(half x) const {
    half zero = __float2half(0);
    return (x > zero) ? x : __float2half(alpha) * x;
  }
  const float alpha;
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
struct SoftplusFunctor<half> {
  OF_DEVICE_FUNC explicit SoftplusFunctor(float beta, float threshold)
      : beta(beta), threshold(threshold), float_functor(SoftplusFunctor<float>(beta, threshold)) {}
  __device__ half operator()(half x) const { return __float2half(float_functor(__half2float(x))); }
  const float beta;
  const float threshold;
  SoftplusFunctor<float> float_functor;
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
struct CeluFunctor<half> {
  OF_DEVICE_FUNC explicit CeluFunctor(float alpha)
      : alpha(alpha), float_functor(CeluFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  const float alpha;
  CeluFunctor<float> float_functor;
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
struct HardShrinkFunctor<half> {
  OF_DEVICE_FUNC explicit HardShrinkFunctor(float lambd)
      : lambd(lambd), float_functor(HardShrinkFunctor<float>(lambd)) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  const float lambd;
  HardShrinkFunctor<float> float_functor;
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

template<>
struct ThresholdFunctor<half> {
  OF_DEVICE_FUNC explicit ThresholdFunctor(float threshold, float value)
      : threshold(threshold),
        value(value),
        float_functor(ThresholdFunctor<float>(threshold, value)) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  const float threshold;
  const float value;
  ThresholdFunctor<float> float_functor;
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
struct SoftShrinkFunctor<half> {
  OF_DEVICE_FUNC explicit SoftShrinkFunctor(float alpha)
      : alpha(alpha), float_functor(SoftShrinkFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
  const float alpha;
  SoftShrinkFunctor<float> float_functor;
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

#define REGISTER_ACTIVATION_CUDA_KERNEL(dtype)           \
  REGISTER_ELU_KERNEL(DeviceType::kCUDA, dtype);         \
  REGISTER_CELU_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_HARDSWISH_KERNEL(DeviceType::kCUDA, dtype);   \
  REGISTER_HARDSIGMOID_KERNEL(DeviceType::kCUDA, dtype); \
  REGISTER_HARDSHRINK_KERNEL(DeviceType::kCUDA, dtype);  \
  REGISTER_HARDTANH_KERNEL(DeviceType::kCUDA, dtype);    \
  REGISTER_MISH_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_SILU_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_SELU_KERNEL(DeviceType::kCUDA, dtype);        \
  REGISTER_SOFTSHRINK_KERNEL(DeviceType::kCUDA, dtype);  \
  REGISTER_SOFTSIGN_KERNEL(DeviceType::kCUDA, dtype);    \
  REGISTER_LEAKYRELU_KERNEL(DeviceType::kCUDA, dtype);   \
  REGISTER_THRESHOLD_KERNEL(DeviceType::kCUDA, dtype);   \
  REGISTER_SOFTPLUS_KERNEL(DeviceType::kCUDA, dtype);    \
  REGISTER_RELU_BACKWARD_KERNEL(DeviceType::kCUDA, dtype);

namespace {

REGISTER_ACTIVATION_CUDA_KERNEL(half);
REGISTER_ACTIVATION_CUDA_KERNEL(float);
REGISTER_ACTIVATION_CUDA_KERNEL(double);

}  // namespace

}  // namespace oneflow
