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
#ifndef ONEFLOW_USER_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_
#define ONEFLOW_USER_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/ops/math_unary_elementwise_seq.h"
#include "oneflow/core/device/cuda_pseudo_half.h"

#if defined(__CUDACC__)

#include <cuda_fp16.h>
#define MATH_FUNC_F(name, x) name##f(x)
#define MATH_FUNC_D(name, x) name(x)

#else

#include <cmath>
#define MATH_FUNC_F(name, x) std::name(x)
#define MATH_FUNC_D(name, x) std::name(x)

#endif

namespace oneflow {

#define DECLARE_UNARY_FUNCTOR(math_unary_elementwise_type, func_prefix) \
  template<typename T>                                                  \
  struct func_prefix##Functor;

OF_PP_FOR_EACH_TUPLE(DECLARE_UNARY_FUNCTOR, MATH_UNARY_ELEMENTWISE_FUNC_SEQ)

template<typename T>
struct AbsFunctor {
  static OF_DEVICE_FUNC T Forward(const T x) {
    if (x == T(0))
      return T(0);
    else
      return x < T(0) ? -x : x;
  }

  static OF_DEVICE_FUNC T Backward(const T x, const T dy) {
    if (x == T(0))
      return T(0);
    else
      return x < T(0) ? -dy : dy;
  }
};

template<typename T>
struct SignFunctor {
  static OF_DEVICE_FUNC T Forward(const T x) { return (T(0) < x) - (x < T(0)); }

  static OF_DEVICE_FUNC T Backward(const T x, const T dy) { return T(0); }
};

template<>
struct RsqrtFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) {
#if defined(__CUDACC__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
  }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (-1.0f / (2.0f * MATH_FUNC_F(sqrt, x * x * x)));
  }
};

template<>
struct RsqrtFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) {
#if defined(__CUDACC__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
  }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (-1.0 / (2.0 * MATH_FUNC_D(sqrt, x * x * x)));
  }
};

// float version

template<>
struct AcosFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(acos, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * -RsqrtFunctor<float>::Forward(1.0f - x * x);
  }
};

template<>
struct AcoshFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(acosh, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * RsqrtFunctor<float>::Forward(x * x - 1.0f);
  }
};

template<>
struct AsinFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(asin, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * RsqrtFunctor<float>::Forward(1.0f - x * x);
  }
};

template<>
struct AsinhFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(asinh, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * RsqrtFunctor<float>::Forward(1.0f + x * x);
  }
};

template<>
struct AtanFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(atan, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (1.0f / (1.0f + x * x));
  }
};

template<>
struct AtanhFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(atanh, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (1.0f / (1.0f - x * x));
  }
};

template<>
struct NotEqualZeroFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return x != 0; }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct CeilFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(ceil, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct CosFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(cos, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (-MATH_FUNC_F(sin, x));
  }
};

template<>
struct CoshFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(cosh, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(sinh, x);
  }
};

template<>
struct ErfFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(erf, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * 2.0f * RsqrtFunctor<float>::Forward(M_PI) * expf(-x * x);
  }
};

template<>
struct ErfcFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(erfc, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * -2.0f * RsqrtFunctor<float>::Forward(M_PI) * expf(-x * x);
  }
};

template<>
struct ExpFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(exp, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(exp, x);
  }
};

template<>
struct Expm1Functor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(expm1, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(exp, x);
  }
};

template<>
struct FloorFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(floor, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct LgammaFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(lgamma, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    // TODO(chengcheng): return: dy * digamma(x)
    assert(false);
    return 0.0f;
  }
};

template<>
struct LogFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(log, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return dy * (1.0f / x); }
};

template<>
struct Log2Functor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(log2, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (1.0f / (x * MATH_FUNC_F(log, 2.0f)));
  }
};

template<>
struct Log1pFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(log1p, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (1.0f / (x + 1.0f));
  }
};

template<>
struct LogSigmoidFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) {
    return -MATH_FUNC_F(log, (1.0f + MATH_FUNC_F(exp, -x)));
  }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (1.0f / (MATH_FUNC_F(exp, x) + 1.0f));
  }
};

template<>
struct NegativeFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return -x; }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return -dy; }
};

template<>
struct ReciprocalFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return 1.0f / x; }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (-1.0f / (x * x));
  }
};

template<>
struct ReciprocalNoNanFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) {
    if (fabsf(x) <= 0.0f) { return 0.0f; }
    return 1.0f / x;
  }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    if (fabsf(x) <= 0.0f) { return 0.0f; }
    return dy * (-1.0f / (x * x));
  }
};

template<>
struct RintFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(rint, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct RoundFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(nearbyint, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct SigmoidFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) {
    return 1.0f / (1.0f + MATH_FUNC_F(exp, -x));
  }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    float y = 1.0f / (1.0f + MATH_FUNC_F(exp, -x));
    return dy * (y * (1.0f - y));
  }
};

template<>
struct SinFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(sin, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(cos, x);
  }
};

template<>
struct SinhFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(sinh, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(cosh, x);
  }
};

template<>
struct SqrtFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(sqrt, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * 0.5f / MATH_FUNC_F(sqrt, x);
  }
};

template<>
struct SquareFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return x * x; }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) { return dy * 2.0f * x; }
};

template<>
struct TanFunctor<float> {
  static OF_DEVICE_FUNC float Forward(const float x) { return MATH_FUNC_F(tan, x); }

  static OF_DEVICE_FUNC float Backward(const float x, const float dy) {
    return dy * (1.0f / (MATH_FUNC_F(cos, x) * MATH_FUNC_F(cos, x)));
  }
};

// double version

template<>
struct AcosFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(acos, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * -RsqrtFunctor<double>::Forward(1.0 - x * x);
  }
};

template<>
struct AcoshFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(acosh, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * -RsqrtFunctor<double>::Forward(x * x - 1.0);
  }
};

template<>
struct AsinFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(asin, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * RsqrtFunctor<double>::Forward(1.0 - x * x);
  }
};

template<>
struct AsinhFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(asinh, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * RsqrtFunctor<double>::Forward(1.0 + x * x);
  }
};

template<>
struct AtanFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(atan, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (1.0 / (1.0 + x * x));
  }
};

template<>
struct AtanhFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(atanh, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (1.0 / (1.0 - x * x));
  }
};

template<>
struct NotEqualZeroFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return x != 0; }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return 0.0f; }
};

template<>
struct CeilFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(ceil, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct CosFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(cos, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (-MATH_FUNC_D(sin, x));
  }
};

template<>
struct CoshFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(cosh, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(sinh, x);
  }
};

template<>
struct ErfFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(erf, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * 2.0 * RsqrtFunctor<double>::Forward(M_PI) * expf(-x * x);
  }
};

template<>
struct ErfcFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(erfc, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * -2.0 * RsqrtFunctor<double>::Forward(M_PI) * expf(-x * x);
  }
};

template<>
struct ExpFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(exp, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(exp, x);
  }
};

template<>
struct Expm1Functor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(expm1, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(exp, x);
  }
};

template<>
struct FloorFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(floor, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct LgammaFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(lgamma, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    // TODO(chengcheng): return: dy * digamma(x)
    assert(false);
    return 0.0;
  }
};

template<>
struct LogFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(log, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return dy * (1.0 / x); }
};

template<>
struct Log2Functor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(log2, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (1.0 / (x * MATH_FUNC_D(log, 2.0)));
  }
};

template<>
struct Log1pFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(log1p, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (1.0 / (x + 1.0));
  }
};

template<>
struct LogSigmoidFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) {
    return -MATH_FUNC_D(log, (1.0 + MATH_FUNC_D(exp, -x)));
  }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (1.0 / (MATH_FUNC_D(exp, x) + 1.0));
  }
};

template<>
struct NegativeFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return -x; }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return -dy; }
};

template<>
struct ReciprocalFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return 1.0 / x; }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (-1.0 / (x * x));
  }
};

template<>
struct ReciprocalNoNanFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) {
    if (fabs(x) <= 0.0) { return 0.0; }
    return 1.0 / x;
  }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    if (fabs(x) <= 0.0) { return 0.0; }
    return dy * (-1.0 / (x * x));
  }
};

template<>
struct RintFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(rint, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct RoundFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(nearbyint, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct SigmoidFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) {
    return 1.0 / (1.0 + MATH_FUNC_D(exp, -x));
  }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    double y = 1.0 / (1.0 + MATH_FUNC_D(exp, -x));
    return dy * (y * (1.0 - y));
  }
};

template<>
struct SinFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(sin, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(cos, x);
  }
};

template<>
struct SinhFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(sinh, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(cosh, x);
  }
};

template<>
struct SqrtFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(sqrt, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (double)0.5 / MATH_FUNC_D(sqrt, x);
  }
};

template<>
struct SquareFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return x * x; }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) { return dy * 2.0 * x; }
};

template<>
struct TanFunctor<double> {
  static OF_DEVICE_FUNC double Forward(const double x) { return MATH_FUNC_D(tan, x); }

  static OF_DEVICE_FUNC double Backward(const double x, const double dy) {
    return dy * (1.0 / (MATH_FUNC_D(cos, x) * MATH_FUNC_D(cos, x)));
  }
};

#if defined(__CUDACC__)
// half version

#define OF_HALF_FUNC __device__ __forceinline__

#define MATH_FUNC_H(name, x) __float2half(name##f(__half2float(x)))
#define HALF_VAL_HALF __float2half(0.5f)
#define HALF_VAL_TWO __float2half(2.0f)
#define HALF_VAL_2RSQRT_PI __float2half(1.1283791671f)

template<>
struct AbsFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) {
    return __hlt(x, GetZeroVal<half>()) ? __hneg(x) : x;
  }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hlt(x, GetZeroVal<half>()) ? __hneg(dy) : dy;
  }
};

template<>
struct AcosFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(acos, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hneg(hrsqrt(__hsub(GetOneVal<half>(), __hmul(x, x)))));
  }
};

template<>
struct AcoshFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(acosh, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, hrsqrt(__hsub(__hmul(x, x), GetOneVal<half>())));
  }
};

template<>
struct AsinFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(asin, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, hrsqrt(__hsub(GetOneVal<half>(), __hmul(x, x))));
  }
};

template<>
struct AsinhFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(asinh, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, hrsqrt(__hadd(GetOneVal<half>(), __hmul(x, x))));
  }
};

template<>
struct AtanFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(atan, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hdiv(GetOneVal<half>(), __hadd(GetOneVal<half>(), __hmul(x, x))));
  }
};

template<>
struct AtanhFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(atanh, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hdiv(GetOneVal<half>(), __hsub(GetOneVal<half>(), __hmul(x, x))));
  }
};

template<>
struct CeilFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hceil(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return GetZeroVal<half>(); }
};

template<>
struct NotEqualZeroFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return x != static_cast<half>(0.0); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return GetZeroVal<half>(); }
};

template<>
struct CosFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hcos(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hneg(hsin(x)));
  }
};

template<>
struct CoshFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(cosh, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, MATH_FUNC_H(sinh, x));
  }
};

template<>
struct ErfFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(erf, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hmul(HALF_VAL_2RSQRT_PI, hexp(__hmul(__hneg(x), x))));
  }
};

template<>
struct ErfcFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(erfc, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hneg(__hmul(HALF_VAL_2RSQRT_PI, hexp(__hmul(__hneg(x), x)))));
  }
};

template<>
struct ExpFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hexp(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return __hmul(dy, hexp(x)); }
};

template<>
struct Expm1Functor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(expm1, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return __hmul(dy, hexp(x)); }
};

template<>
struct FloorFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hfloor(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return GetZeroVal<half>(); }
};

template<>
struct LgammaFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(lgamma, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    // TODO(chengcheng): return: dy * digamma(x)
    assert(false);
    return GetZeroVal<half>();
  }
};

template<>
struct LogFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hlog(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return __hmul(dy, hrcp(x)); }
};

template<>
struct Log2Functor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hlog2(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, hrcp(__hmul(x, hlog(HALF_VAL_TWO))));
  }
};

template<>
struct Log1pFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(log1p, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, hrcp(__hadd(x, GetOneVal<half>())));
  }
};

template<>
struct LogSigmoidFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) {
    return __hneg(hlog(__hadd(GetOneVal<half>(), hexp(__hneg(x)))));
  }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, hrcp(__hadd(hexp(x), GetOneVal<half>())));
  }
};

template<>
struct NegativeFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return __hneg(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return __hneg(dy); }
};

template<>
struct ReciprocalFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hrcp(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hneg(hrcp(__hmul(x, x))));
  }
};

template<>
struct ReciprocalNoNanFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) {
    if (__heq(GetZeroVal<half>(), x)) { return GetZeroVal<half>(); }
    return hrcp(x);
  }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    if (__heq(GetZeroVal<half>(), x)) { return GetZeroVal<half>(); }
    return __hmul(dy, __hneg(hrcp(__hmul(x, x))));
  }
};

template<>
struct RintFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hrint(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return GetZeroVal<half>(); }
};

template<>
struct RoundFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(nearbyint, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return GetZeroVal<half>(); }
};

template<>
struct RsqrtFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hrsqrt(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hneg(hrcp(__hmul(HALF_VAL_TWO, hsqrt(__hmul(x, __hmul(x, x)))))));
  }
};

template<>
struct SigmoidFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) {
    return hrcp(__hadd(GetOneVal<half>(), hexp(__hneg(x))));
  }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    half y = hrcp(__hadd(GetOneVal<half>(), hexp(__hneg(x))));
    return __hmul(dy, __hmul(y, __hsub(GetOneVal<half>(), y)));
  }
};

template<>
struct SignFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) {
    if (__hgt(x, GetZeroVal<half>())) { return GetOneVal<half>(); }
    if (__hlt(x, GetZeroVal<half>())) { return __hneg(GetOneVal<half>()); }
    return GetZeroVal<half>();
  }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return GetZeroVal<half>(); }
};

template<>
struct SinFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hsin(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) { return __hmul(dy, hcos(x)); }
};

template<>
struct SinhFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return MATH_FUNC_H(sinh, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, MATH_FUNC_H(cosh, x));
  }
};

template<>
struct SqrtFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return hsqrt(x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hdiv(HALF_VAL_HALF, hsqrt(x)));
  }
};

template<>
struct SquareFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return __hmul(x, x); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, __hmul(HALF_VAL_TWO, x));
  }
};

template<>
struct TanFunctor<half> {
  static OF_HALF_FUNC half Forward(const half x) { return __hdiv(hsin(x), hcos(x)); }

  static OF_HALF_FUNC half Backward(const half x, const half dy) {
    return __hmul(dy, hrcp(__hmul(hcos(x), hcos(x))));
  }
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_
