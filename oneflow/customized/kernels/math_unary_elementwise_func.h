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
  static OF_DEVICE_FUNC const float Forward(const float x) { return MATH_FUNC_F(round, x); }

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

/*

#define MATH_UNARY_ELEMENTWISE_FUNC_SEQ                                                   \
  OF_PP_MAKE_TUPLE_SEQ("Abs", fabsf, AbsBw<float>)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Acos", acosf, AcosBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Acosh", acoshf, AcoshBwFloat)                                     \
  OF_PP_MAKE_TUPLE_SEQ("Asin", asinf, AsinBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Asinh", asinhf, AsinhBwFloat)                                     \
  OF_PP_MAKE_TUPLE_SEQ("Atan", atanf, AtanBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Atanh", atanhf, AtanhBwFloat)                                     \
  OF_PP_MAKE_TUPLE_SEQ("Ceil", ceilf, CeilBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Cos", cosf, CosBwFloat)                                           \
  OF_PP_MAKE_TUPLE_SEQ("Cosh", coshf, CoshBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Erf", erff, ErfBwFloat)                                           \
  OF_PP_MAKE_TUPLE_SEQ("Erfc", erfcf, ErfcBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Exp", expf, ExpBwFloat)                                           \
  OF_PP_MAKE_TUPLE_SEQ("Expm1", expm1f, Expm1BwFloat)                                     \
  OF_PP_MAKE_TUPLE_SEQ("Floor", floorf, FloorBwFloat)                                     \
  OF_PP_MAKE_TUPLE_SEQ("Lgamma", lgammaf, LgammaBwFloat)                                  \
  OF_PP_MAKE_TUPLE_SEQ("Log", logf, LogBwFloat)                                           \
  OF_PP_MAKE_TUPLE_SEQ("Log1p", log1pf, Log1pBwFloat)                                     \
  OF_PP_MAKE_TUPLE_SEQ("LogSigmoid", LogSigmoidFwFloat, LogSigmoidBwFloat)                \
  OF_PP_MAKE_TUPLE_SEQ("Negative", NegativeFwFloat, NegativeBwFloat)                      \
  OF_PP_MAKE_TUPLE_SEQ("Reciprocal", ReciprocalFwFloat, ReciprocalBwFloat)                \
  OF_PP_MAKE_TUPLE_SEQ("ReciprocalNoNan", ReciprocalNoNanFwFloat, ReciprocalNoNanBwFloat) \
  OF_PP_MAKE_TUPLE_SEQ("Rint", rintf, RintBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Round", nearbyintf, RoundBwFloat)                                 \
  OF_PP_MAKE_TUPLE_SEQ("Rsqrt", rsqrtf, RsqrtBwFloat)                                     \
  OF_PP_MAKE_TUPLE_SEQ("Sigmoid", SigmoidFwFloat, SigmoidBwFloat)                         \
  OF_PP_MAKE_TUPLE_SEQ("Sign", SignFwFloat, SignBwFloat)                                  \
  OF_PP_MAKE_TUPLE_SEQ("Sin", sinf, SinBwFloat)                                           \
  OF_PP_MAKE_TUPLE_SEQ("Sinh", sinhf, SinhBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Softplus", SoftplusFwFloat, SoftplusBwFloat)                      \
  OF_PP_MAKE_TUPLE_SEQ("Sqrt", sqrtf, SqrtBwFloat)                                        \
  OF_PP_MAKE_TUPLE_SEQ("Square", SquareFwFloat, SquareBwFloat)                            \
  OF_PP_MAKE_TUPLE_SEQ("Tan", tanf, TanBwFloat)                                           \
  OF_PP_MAKE_TUPLE_SEQ("Tanh", tanhf, TanhBwFloat)
*/

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_
