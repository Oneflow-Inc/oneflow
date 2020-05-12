#ifndef ONEFLOW_CUSTOMIZED_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/customized/ops/math_unary_elementwise_seq.h"

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
  static OF_DEVICE_FUNC const T Forward(const T x) { return x < 0 ? -x : x; }

  static OF_DEVICE_FUNC const T Backward(const T x, const T dy) { return x < 0 ? -dy : dy; }
};

// float version

template<>
struct AcosFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(acos, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
#if defined(__CUDACC__)
    return dy * (-rsqrtf(1.0f - x * x));
#else
    return dy * (-1.0f / std::sqrt(1.0f - x * x));
#endif
  }
};

template<>
struct AcoshFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(acosh, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
#if defined(__CUDACC__)
    return dy * (rsqrtf(x * x - 1.0f));
#else
    return dy * (1.0f / std::sqrt(x * x - 1.0f));
#endif
  }
};

template<>
struct AsinFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(asin, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
#if defined(__CUDACC__)
    return dy * (rsqrtf(1.0f - x * x));
#else
    return dy * (1.0f / std::sqrt(1.0f - x * x));
#endif
  }
};

template<>
struct AsinhFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(asinh, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
#if defined(__CUDACC__)
    return dy * (rsqrtf(1.0f + x * x));
#else
    return dy * (1.0f / std::sqrt(1.0f + x * x));
#endif
  }
};

template<>
struct AtanFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(atan, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (1.0f / (1.0f + x * x));
  }
};

template<>
struct AtanhFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(atanh, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (1.0f / (1.0f - x * x));
  }
};

template<>
struct CeilFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(ceil, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct CosFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(cos, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (-MATH_FUNC_F(sin, x));
  }
};

template<>
struct CoshFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(cosh, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (MATH_FUNC_F(exp, x) + MATH_FUNC_F(exp, -x)) / 2.0f;
  }
};

template<>
struct ErfFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(erf, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
#if defined(__CUDACC__)
    return dy * 2.0f * rsqrtf(M_PI) * expf(-x * x);
#else
    return dy * 2.0f * (1.0f / std::sqrt(M_PI)) * std::exp(-x * x);
#endif
  }
};

template<>
struct ErfcFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(erfc, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
#if defined(__CUDACC__)
    return dy * -2.0f * rsqrtf(M_PI) * expf(-x * x);
#else
    return dy * -2.0f * (1.0f / std::sqrt(M_PI)) * std::exp(-x * x);
#endif
  }
};

template<>
struct ExpFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(exp, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(exp, x);
  }
};

template<>
struct Expm1Functor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(expm1, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(exp, x);
  }
};

template<>
struct FloorFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(floor, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct LgammaFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(lgamma, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    // TODO(chengcheng): return: dy * digamma(x)
    assert(false);
    return 0.0f;
  }
};

template<>
struct LogFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(log, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (1.0f / x);
  }
};

template<>
struct Log1pFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(log1p, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (1.0f / (x + 1.0f));
  }
};

template<>
struct LogSigmoidFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) {
    return MATH_FUNC_F(log, (1.0f / (1.0f + MATH_FUNC_F(exp, -x))));
  }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (1.0f / (MATH_FUNC_F(exp, x) + 1.0f));
  }
};

template<>
struct NegativeFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return -x; }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) { return -dy; }
};

template<>
struct ReciprocalFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return 1.0f / x; }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (-1.0f / (x * x));
  }
};

template<>
struct ReciprocalNoNanFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) {
    if (fabsf(x) <= 0.0f) { return 0.0f; }
    return 1.0f / x;
  }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    if (fabsf(x) <= 0.0f) { return 0.0f; }
    return dy * (-1.0f / (x * x));
  }
};

template<>
struct RintFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(rint, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct RoundFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(nearbyint, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct RsqrtFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) {
#if defined(__CUDACC__)
    return rsqrtf(x);
#else
    return 1.0f / std::sqrt(x);
#endif
  }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (-1.0f / (2.0f * MATH_FUNC_F(sqrt, x * x * x)));
  }
};

template<>
struct SigmoidFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) {
    return 1.0f / (1.0f + MATH_FUNC_F(exp, -x));
  }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    float y = 1.0f / (1.0f + MATH_FUNC_F(exp, -x));
    return dy * (y * (1.0f - y));
  }
};

template<>
struct SignFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) {
    if (x > 0.0f) { return 1.0f; }
    if (x < 0.0f) { return -1.0f; }
    return 0.0f;
  }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) { return 0.0f; }
};

template<>
struct SinFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(sin, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(cos, x);
  }
};

template<>
struct SinhFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(sinh, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(cosh, x);
  }
};

template<>
struct SoftplusFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) {
    return MATH_FUNC_F(log, (1.0f + MATH_FUNC_F(exp, x)));
  }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * MATH_FUNC_F(exp, x) / (MATH_FUNC_F(exp, x) + 1.0f);
  }
};

template<>
struct SqrtFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(sqrt, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * 0.5f / MATH_FUNC_F(sqrt, x);
  }
};

template<>
struct SquareFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return x * x; }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * 2.0f * x;
  }
};

template<>
struct TanFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(tan, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (1.0f / (MATH_FUNC_F(cos, x) * MATH_FUNC_F(cos, x)));
  }
};

template<>
struct TanhFunctor<float> {
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(tanh, x); }

  static OF_DEVICE_FUNC const float Backward(const float x, const float dy) {
    return dy * (1.0f - MATH_FUNC_F(tanh, x) * MATH_FUNC_F(tanh, x));
  }
};

// double version

template<>
struct AcosFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(acos, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
#if defined(__CUDACC__)
    return dy * (-rsqrt(1.0 - x * x));
#else
    return dy * (-1.0 / std::sqrt(1.0 - x * x));
#endif
  }
};

template<>
struct AcoshFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(acosh, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
#if defined(__CUDACC__)
    return dy * (rsqrt(x * x - 1.0));
#else
    return dy * (1.0 / std::sqrt(x * x - 1.0));
#endif
  }
};

template<>
struct AsinFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(asin, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
#if defined(__CUDACC__)
    return dy * (rsqrt(1.0 - x * x));
#else
    return dy * (1.0 / std::sqrt(1.0 - x * x));
#endif
  }
};

template<>
struct AsinhFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(asinh, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
#if defined(__CUDACC__)
    return dy * (rsqrt(1.0 + x * x));
#else
    return dy * (1.0 / std::sqrt(1.0 + x * x));
#endif
  }
};

template<>
struct AtanFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(atan, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (1.0 / (1.0 + x * x));
  }
};

template<>
struct AtanhFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(atanh, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (1.0 / (1.0 - x * x));
  }
};

template<>
struct CeilFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(ceil, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct CosFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(cos, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (-MATH_FUNC_D(sin, x));
  }
};

template<>
struct CoshFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(cosh, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (MATH_FUNC_D(exp, x) + MATH_FUNC_D(exp, -x)) / 2.0;
  }
};

template<>
struct ErfFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(erf, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
#if defined(__CUDACC__)
    return dy * 2.0 * rsqrt(M_PI) * exp(-x * x);
#else
    return dy * 2.0 * (1.0 / std::sqrt(M_PI)) * std::exp(-x * x);
#endif
  }
};

template<>
struct ErfcFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(erfc, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
#if defined(__CUDACC__)
    return dy * -2.0 * rsqrt(M_PI) * exp(-x * x);
#else
    return dy * -2.0 * (1.0 / std::sqrt(M_PI)) * std::exp(-x * x);
#endif
  }
};

template<>
struct ExpFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(exp, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(exp, x);
  }
};

template<>
struct Expm1Functor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(expm1, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(exp, x);
  }
};

template<>
struct FloorFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(floor, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct LgammaFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(lgamma, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    // TODO(chengcheng): return: dy * digamma(x)
    assert(false);
    return 0.0;
  }
};

template<>
struct LogFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(log, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (1.0 / x);
  }
};

template<>
struct Log1pFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(log1p, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (1.0 / (x + 1.0));
  }
};

template<>
struct LogSigmoidFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) {
    return MATH_FUNC_D(log, (1.0 / (1.0 + MATH_FUNC_D(exp, -x))));
  }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (1.0 / (MATH_FUNC_D(exp, x) + 1.0));
  }
};

template<>
struct NegativeFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return -x; }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) { return -dy; }
};

template<>
struct ReciprocalFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return 1.0 / x; }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (-1.0 / (x * x));
  }
};

template<>
struct ReciprocalNoNanFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) {
    if (fabs(x) <= 0.0) { return 0.0; }
    return 1.0 / x;
  }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    if (fabs(x) <= 0.0) { return 0.0; }
    return dy * (-1.0 / (x * x));
  }
};

template<>
struct RintFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(rint, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct RoundFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(nearbyint, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct RsqrtFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) {
#if defined(__CUDACC__)
    return rsqrt(x);
#else
    return 1.0 / std::sqrt(x);
#endif
  }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (-1.0 / (2.0 * MATH_FUNC_D(sqrt, x * x * x)));
  }
};

template<>
struct SigmoidFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) {
    return 1.0 / (1.0 + MATH_FUNC_D(exp, -x));
  }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    double y = 1.0 / (1.0 + MATH_FUNC_D(exp, -x));
    return dy * (y * (1.0 - y));
  }
};

template<>
struct SignFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) {
    if (x > 0.0) { return 1.0; }
    if (x < 0.0) { return -1.0; }
    return 0.0;
  }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) { return 0.0; }
};

template<>
struct SinFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(sin, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(cos, x);
  }
};

template<>
struct SinhFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(sinh, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(cosh, x);
  }
};

template<>
struct SoftplusFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) {
    return MATH_FUNC_D(log, (1.0 + MATH_FUNC_D(exp, x)));
  }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * MATH_FUNC_D(exp, x) / (MATH_FUNC_D(exp, x) + 1.0);
  }
};

template<>
struct SqrtFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(sqrt, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * 0.5 / MATH_FUNC_D(sqrt, x);
  }
};

template<>
struct SquareFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return x * x; }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * 2.0 * x;
  }
};

template<>
struct TanFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(tan, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (1.0 / (MATH_FUNC_D(cos, x) * MATH_FUNC_D(cos, x)));
  }
};

template<>
struct TanhFunctor<double> {
  static OF_DEVICE_FUNC const double Forward(const double x) { return MATH_FUNC_D(tanh, x); }

  static OF_DEVICE_FUNC const double Backward(const double x, const double dy) {
    return dy * (1.0 - MATH_FUNC_D(tanh, x) * MATH_FUNC_D(tanh, x));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_
