#ifndef ONEFLOW_CUSTOMIZED_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
OF_DEVICE_FUNC T AbsBw(T x, T dy) {
  return x < 0 ? -dy : dy;
}

OF_DEVICE_FUNC float AcosBwFloat(float x, float dy) { return dy * (-rsqrtf(1.0f - x * x)); }

OF_DEVICE_FUNC float AcoshBwFloat(float x, float dy) { return dy * (rsqrtf(x * x - 1.0f)); }

OF_DEVICE_FUNC float AsinBwFloat(float x, float dy) { return dy * (rsqrtf(1.0f - x * x)); }

OF_DEVICE_FUNC float AsinhBwFloat(float x, float dy) { return dy * (rsqrtf(1.0f + x * x)); }

OF_DEVICE_FUNC float AtanBwFloat(float x, float dy) { return dy * (1.0f / (1.0f + x * x)); }

OF_DEVICE_FUNC float AtanhBwFloat(float x, float dy) { return dy * (1.0f / (1.0f - x * x)); }

OF_DEVICE_FUNC float CeilBwFloat(float x, float dy) { return 0.0f; }

OF_DEVICE_FUNC float CosBwFloat(float x, float dy) { return dy * (-sinf(x)); }

OF_DEVICE_FUNC float CoshBwFloat(float x, float dy) { return dy * (expf(x) + expf(-x)) / 2.0f; }

OF_DEVICE_FUNC float ErfBwFloat(float x, float dy) {
  return dy * 2.0f * rsqrtf(M_PI) * expf(-x * x);
}

OF_DEVICE_FUNC float ErfcBwFloat(float x, float dy) {
  return dy * -2.0f * rsqrtf(M_PI) * expf(-x * x);
}

OF_DEVICE_FUNC float ExpBwFloat(float x, float dy) { return dy * expf(x); }

OF_DEVICE_FUNC float Expm1BwFloat(float x, float dy) { return dy * expf(x); }

OF_DEVICE_FUNC float FloorBwFloat(float x, float dy) { return 0.0f; }

OF_DEVICE_FUNC int8_t IsFinite(float x) {
  return isfinite(x) ? 1 : 0;
}  // use int8 1 as true; int8 0 as false

OF_DEVICE_FUNC int8_t IsInf(float x) { return isinf(x) ? 1 : 0; }

OF_DEVICE_FUNC int8_t IsNaN(float x) { return isnan(x) ? 1 : 0; }

OF_DEVICE_FUNC float LgammaBwFloat(float x, float dy) {
  // TODO(chengcheng): return: dy * digamma(x)
  assert(false);
  return 0.0f;
}

OF_DEVICE_FUNC float LogBwFloat(float x, float dy) { return dy * (1.0f / x); }

OF_DEVICE_FUNC float Log1pBwFloat(float x, float dy) { return dy * (1.0f / (x + 1.0f)); }

OF_DEVICE_FUNC float LogSigmoidFwFloat(float x) { return logf(1.0f / (1.0f + expf(-x))); }

OF_DEVICE_FUNC float LogSigmoidBwFloat(float x, float dy) { return dy * (1.0f / (expf(x) + 1.0f)); }

OF_DEVICE_FUNC float NegativeFwFloat(float x) { return -x; }

OF_DEVICE_FUNC float NegativeBwFloat(float x, float dy) { return -dy; }

OF_DEVICE_FUNC float ReciprocalFwFloat(float x) { return 1.0f / x; }

OF_DEVICE_FUNC float ReciprocalBwFloat(float x, float dy) { return dy * (-1.0f / (x * x)); }

OF_DEVICE_FUNC float ReciprocalNoNanFwFloat(float x) {
  if (fabsf(x) <= 0.0f) { return 0.0f; }
  return 1.0f / x;
}

OF_DEVICE_FUNC float ReciprocalNoNanBwFloat(float x, float dy) {
  if (fabsf(x) <= 0.0f) { return 0.0f; }
  return dy * (-1.0f / (x * x));
}
OF_DEVICE_FUNC float RintBwFloat(float x, float dy) { return 0.0f; }

OF_DEVICE_FUNC float RoundBwFloat(float x, float dy) { return 0.0f; }

OF_DEVICE_FUNC float RsqrtBwFloat(float x, float dy) {
  return dy * (-1.0f / (2.0f * sqrtf(x * x * x)));
}

OF_DEVICE_FUNC float SigmoidFwFloat(float x) { return 1.0f / (1.0f + expf(-x)); }

OF_DEVICE_FUNC float SigmoidBwFloat(float x, float dy) {
  float y = SigmoidFwFloat(x);
  return dy * (y * (1.0f - y));
}

OF_DEVICE_FUNC float SignFwFloat(float x) {
  if (x > 0.0f) { return 1.0f; }
  if (x < 0.0f) { return -1.0f; }
  return 0.0f;
}

OF_DEVICE_FUNC float SignBwFloat(float x, float dy) { return 0.0f; }

OF_DEVICE_FUNC float SinBwFloat(float x, float dy) { return dy * cosf(x); }

OF_DEVICE_FUNC float SinhBwFloat(float x, float dy) { return dy * expf(x) - expf(-x) * 0.5f; }

OF_DEVICE_FUNC float SoftplusFwFloat(float x) { return logf(expf(x) + 1.0f); }

OF_DEVICE_FUNC float SoftplusBwFloat(float x, float dy) { return dy * expf(x) / (expf(x) + 1); }

OF_DEVICE_FUNC float SqrtBwFloat(float x, float dy) { return dy * 0.5f * rsqrtf(x); }

OF_DEVICE_FUNC float SquareFwFloat(float x) { return x * x; }

OF_DEVICE_FUNC float SquareBwFloat(float x, float dy) { return dy * 2.0f * x; }

OF_DEVICE_FUNC float TanBwFloat(float x, float dy) { return dy * (1.0f / (cosf(x) * cosf(x))); }

OF_DEVICE_FUNC float TanhBwFloat(float x, float dy) { return dy * sinhf(x) / coshf(x); }

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

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_MATH_UNARY_ELEMENTWISE_FUNC_H_
