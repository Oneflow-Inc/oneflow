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
#ifndef ONEFLOW_USER_KERNELS_MATH_BINARY_ELEMENTWISE_FUNC_H_
#define ONEFLOW_USER_KERNELS_MATH_BINARY_ELEMENTWISE_FUNC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/ops/math_binary_elementwise_seq.h"
#include "oneflow/core/device/cuda_pseudo_half.h"

#if defined(__CUDACC__)

#include <cuda_fp16.h>
#define MATH_FUNC(name) name

#else

#include <cmath>
#define MATH_FUNC(name) std::name

#endif

namespace oneflow {

#define DECLARE_BINARY_FUNCTOR(math_binary_elementwise_type, func_prefix) \
  template<typename T>                                                    \
  struct func_prefix##Functor;

OF_PP_FOR_EACH_TUPLE(DECLARE_BINARY_FUNCTOR, MATH_BINARY_ELEMENTWISE_FUNC_SEQ)

template<typename T>
struct PowFunctor {
  static OF_DEVICE_FUNC const T Forward(const T x, const T y) { return MATH_FUNC(pow)(x, y); }

  static OF_DEVICE_FUNC const T BackwardXGrad(const T x, const T y, const T dz) {
    return dz * y * (MATH_FUNC(pow)(x, y - T(1)));
  }

  static OF_DEVICE_FUNC const T BackwardYGrad(const T x, const T y, const T dz) {
    if (x > T(0)) {
      return dz * MATH_FUNC(log)(x) * (MATH_FUNC(pow)(x, y));
    } else {
      return T(0);
    }
  }
};

template<typename T>
struct Atan2Functor {
  static OF_DEVICE_FUNC const T Forward(const T x, const T y) { return MATH_FUNC(atan2)(x, y); }

  static OF_DEVICE_FUNC const T BackwardXGrad(const T x, const T y, const T dz) {
    return dz * (y / (x * x + y * y));
  }

  static OF_DEVICE_FUNC const T BackwardYGrad(const T x, const T y, const T dz) {
    return dz * -x / (y * y + x * x);
  }
};

template<typename T>
struct FloorDivFunctor {
  static OF_DEVICE_FUNC const T Forward(const T x, const T y) {
#if defined(__CUDACC__)
    return floor(fdividef(x, y));
#else
    return std::floor(x / y);
#endif
  }

  static OF_DEVICE_FUNC const T BackwardXGrad(const T x, const T y, const T dz) { return T(0); }

  static OF_DEVICE_FUNC const T BackwardYGrad(const T x, const T y, const T dz) { return T(0); }
};

template<typename T>
struct TruncDivFunctor {
  static OF_DEVICE_FUNC const T Forward(const T x, const T y) {
#if defined(__CUDACC__)
    return trunc(fdividef(x, y));
#else
    return std::trunc(x / y);
#endif
  }

  static OF_DEVICE_FUNC const T BackwardXGrad(const T x, const T y, const T dz) { return T(0); }

  static OF_DEVICE_FUNC const T BackwardYGrad(const T x, const T y, const T dz) { return T(0); }
};

template<typename T>
struct XdivyFunctor {
  static OF_DEVICE_FUNC const T Forward(const T x, const T y) {
    if (T(0) == x) {
      return T(0);
    } else {
      return x / y;
    }
  }

  static OF_DEVICE_FUNC const T BackwardXGrad(const T x, const T y, const T dz) {
    if (T(0) == x || T(0) == dz) {
      return T(0);
    } else {
      return dz / y;
    }
  }

  static OF_DEVICE_FUNC const T BackwardYGrad(const T x, const T y, const T dz) {
    return dz * XdivyFunctor<T>::Forward((-x), (y * y));
  }
};

template<typename T>
struct XlogyFunctor {
  static OF_DEVICE_FUNC const T Forward(const T x, const T y) {
    if (T(0) == x) {
      return T(0);
    } else {
      return x * MATH_FUNC(log)(y);
    }
  }

  static OF_DEVICE_FUNC const T BackwardXGrad(const T x, const T y, const T dz) {
    if (T(0) == x || T(0) == dz) {
      return T(0);
    } else {
      return dz * MATH_FUNC(log)(y);
    }
  }

  static OF_DEVICE_FUNC const T BackwardYGrad(const T x, const T y, const T dz) {
    return dz * XdivyFunctor<T>::Forward(x, y);
  }
};

#if defined(__CUDACC__)
// half version

#define OF_HALF_FUNC __device__ __forceinline__

#define MATH_FUNC_H_FW(name) __float2half(name(__half2float(x), __half2float(y)))
#define MATH_FUNC_H_BW(name) __float2half(name(__half2float(x), __half2float(y), __half2float(dz)))

template<>
struct PowFunctor<half> {
  static OF_HALF_FUNC const half Forward(const half x, const half y) {
    return MATH_FUNC_H_FW(PowFunctor<float>::Forward);
  }

  static OF_HALF_FUNC const half BackwardXGrad(const half x, const half y, const half dz) {
    return MATH_FUNC_H_BW(PowFunctor<float>::BackwardXGrad);
  }

  static OF_HALF_FUNC const half BackwardYGrad(const half x, const half y, const half dz) {
    return MATH_FUNC_H_BW(PowFunctor<float>::BackwardYGrad);
  }
};

template<>
struct Atan2Functor<half> {
  static OF_HALF_FUNC const half Forward(const half x, const half y) {
    return MATH_FUNC_H_FW(Atan2Functor<float>::Forward);
  }

  static OF_HALF_FUNC const half BackwardXGrad(const half x, const half y, const half dz) {
    return __hmul(dz, __hdiv(y, __hadd(__hmul(y, y), __hmul(x, x))));
  }

  static OF_HALF_FUNC const half BackwardYGrad(const half x, const half y, const half dz) {
    return __hmul(dz, __hdiv(__hneg(x), __hadd(__hmul(y, y), __hmul(x, x))));
  }
};

template<>
struct FloorDivFunctor<half> {
  static OF_HALF_FUNC const half Forward(const half x, const half y) {
    return hfloor(__hdiv(x, y));
  }

  static OF_HALF_FUNC const half BackwardXGrad(const half x, const half y, const half dz) {
    return GetZeroVal<half>();
  }

  static OF_HALF_FUNC const half BackwardYGrad(const half x, const half y, const half dz) {
    return GetZeroVal<half>();
  }
};

template<>
struct TruncDivFunctor<half> {
  static OF_HALF_FUNC const half Forward(const half x, const half y) {
    return htrunc(__hdiv(x, y));
  }

  static OF_HALF_FUNC const half BackwardXGrad(const half x, const half y, const half dz) {
    return GetZeroVal<half>();
  }

  static OF_HALF_FUNC const half BackwardYGrad(const half x, const half y, const half dz) {
    return GetZeroVal<half>();
  }
};

template<>
struct XdivyFunctor<half> {
  static OF_HALF_FUNC const half Forward(const half x, const half y) {
    if (__heq(GetZeroVal<half>(), x)) {
      return GetZeroVal<half>();
    } else {
      return __hdiv(x, y);
    }
  }

  static OF_HALF_FUNC const half BackwardXGrad(const half x, const half y, const half dz) {
    if (__heq(GetZeroVal<half>(), x)) {
      return GetZeroVal<half>();
    } else {
      return XdivyFunctor<half>::Forward(dz, y);
    }
  }

  static OF_HALF_FUNC const half BackwardYGrad(const half x, const half y, const half dz) {
    return __hmul(dz, XdivyFunctor<half>::Forward(__hneg(x), __hmul(y, y)));
  }
};

template<>
struct XlogyFunctor<half> {
  static OF_HALF_FUNC const half Forward(const half x, const half y) {
    if (__heq(GetZeroVal<half>(), x)) {
      return GetZeroVal<half>();
    } else {
      return __hmul(x, hlog(y));
    }
  }

  static OF_HALF_FUNC const half BackwardXGrad(const half x, const half y, const half dz) {
    if (__heq(GetZeroVal<half>(), x)) {
      return GetZeroVal<half>();
    } else {
      return XlogyFunctor<half>::Forward(dz, y);
    }
  }

  static OF_HALF_FUNC const half BackwardYGrad(const half x, const half y, const half dz) {
    return __hmul(dz, XdivyFunctor<half>::Forward(x, y));
  }
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_MATH_BINARY_ELEMENTWISE_FUNC_H_
