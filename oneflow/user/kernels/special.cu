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
#include "oneflow/core/kernel/util/numerics.cuh"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"
namespace oneflow {
#ifdef WITH_CUDA
namespace {
#define HALF_VAL_2RSQRT_PI __float2half(1.1283791671f)
}  // namespace
template<>
struct EntrFunctor<DeviceType::kCUDA, float> {
  __device__ float operator()(const float x) const {
    if (x > 0.0f) {
      return -x * logf(x);
    } else if (x == 0.0f) {
      return 0.0f;
    } else {
      return -INFINITY;
    }
  }
};

template<>
struct EntrFunctor<DeviceType::kCUDA, double> {
  __device__ double operator()(const double x) const {
    if (x > 0.0) {
      return -x * log(x);
    } else if (x == 0.0) {
      return 0.0;
    } else {
      return -INFINITY;
    }
  }
};

template<>
struct EntrFunctor<DeviceType::kCUDA, half> {
  __device__ half operator()(const half x) const {
    if (__hgt(x, 0.0)) {
      return -x * hlog(x);
    } else if (__heq(x, static_cast<half>(0.0))) {
      return static_cast<half>(0.0);
    } else {
      return static_cast<half>(-INFINITY);
    }
  }
};

template<>
struct EntrGradFunctor<DeviceType::kCUDA, float> {
  __device__ float operator()(const float x, const float dy) const {
    if (x > 0.0f) {
      return (-logf(x) - 1) * dy;
    } else if (x == 0.0f) {
      // inf
      return INFINITY;
    } else {
      return detail::Nan<float>();
    }
  }
};

template<>
struct EntrGradFunctor<DeviceType::kCUDA, double> {
  __device__ double operator()(const double x, const double dy) const {
    if (x > 0.0) {
      return (-log(x) - 1) * dy;
    } else if (x == 0.0) {
      // inf
      return INFINITY;
    } else {
      return detail::Nan<double>();
    }
  }
};

template<>
struct EntrGradFunctor<DeviceType::kCUDA, half> {
  __device__ half operator()(const half x, const half dy) const {
    if (x > static_cast<half>(0.0)) {
      return (-hlog(x) - static_cast<half>(1)) * dy;
    } else if (x == static_cast<half>(0.0)) {
      return INFINITY;
    } else {
      return detail::Nan<double>();
    }
  }
};

template<>
struct ErfFunctor<DeviceType::kCUDA, float> {
  __device__ float operator()(const float x) const { return erff(x); }
};

template<>
struct ErfFunctor<DeviceType::kCUDA, double> {
  __device__ double operator()(const double x) const { return erf(x); }
};

template<>
struct ErfFunctor<DeviceType::kCUDA, half> {
  __device__ half operator()(const half x) const { return __float2half(erf(__half2float(x))); }
};

template<>
struct ErfGradFunctor<DeviceType::kCUDA, float> {
  __device__ float operator()(const float x, const float dy) const {
    return dy * 2.0f * expf(-x * x) / sqrtf(x);
  }
};

template<>
struct ErfGradFunctor<DeviceType::kCUDA, double> {
  __device__ double operator()(const double x, const double dy) const {
    return dy * 2.0 * exp(-x * x) / sqrt(x);
  }
};

template<>
struct ErfGradFunctor<DeviceType::kCUDA, half> {
  __device__ half operator()(const half x, const half dy) const {
    return __hmul(dy, __hmul(HALF_VAL_2RSQRT_PI, hexp(__hmul(__hneg(x), x))));
  }
};

template<>
struct ErfcFunctor<DeviceType::kCUDA, float> {
  __device__ float operator()(const float x) const { return erfcf(x); }
};

template<>
struct ErfcFunctor<DeviceType::kCUDA, double> {
  __device__ double operator()(const double x) const { return erfc(x); }
};

template<>
struct ErfcFunctor<DeviceType::kCUDA, half> {
  __device__ half operator()(const half x) const { return __float2half(erfcf(__half2float(x))); }
};

template<>
struct ErfcGradFunctor<DeviceType::kCUDA, float> {
  __device__ float operator()(const float x, const float dy) const {
    // return dy * -2.0f * RsqrtFunctor<float>::Forward(M_PI) * expf(-x * x);
    return 0;
  }
};

template<>
struct ErfcGradFunctor<DeviceType::kCUDA, double> {
  __device__ double operator()(const double x, const double dy) const {
    // return dy * -2.0f * RsqrtFunctor<double>::Forward(M_PI) * exp(-x * x);
    return 0;
  }
};

template<>
struct ErfcGradFunctor<DeviceType::kCUDA, half> {
  __device__ half operator()(const half x, const half dy) const {
    return __hmul(dy, __hneg(__hmul(HALF_VAL_2RSQRT_PI, hexp(__hmul(__hneg(x), x)))));
  }
};

#define REGISTER_SPECIAL_OPS_CUDA_KERNEL(kernel_name, func_prefix)                              \
  REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, DeviceType::kCUDA, float);  \
  REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, DeviceType::kCUDA, double); \
  REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, DeviceType::kCUDA, half);
OF_PP_FOR_EACH_TUPLE(REGISTER_SPECIAL_OPS_CUDA_KERNEL, SPECIAL_UNARY_OPS)
#undef REGISTER_SPECIAL_OPS_CUDA_KERNEL
#endif  // WITH_CUDA
}  // namespace oneflow
