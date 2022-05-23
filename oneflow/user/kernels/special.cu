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
      // -inf
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
      // -inf
      return -INFINITY;
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

#define REGISTER_SPECIAL_OPS_CUDA_KERNEL(kernel_name, func_prefix)                             \
  REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, DeviceType::kCUDA, float); \
  REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, DeviceType::kCUDA, double);
OF_PP_FOR_EACH_TUPLE(REGISTER_SPECIAL_OPS_CUDA_KERNEL, SPECIAL_UNARY_OPS)
#undef REGISTER_SPECIAL_OPS_CUDA_KERNEL
#endif  // WITH_CUDA
}  // namespace oneflow
