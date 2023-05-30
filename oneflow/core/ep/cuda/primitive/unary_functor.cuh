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

#ifndef ONEFLOW_CORE_EP_CUDA_PRIMITIVE_UNARY_FUNCTOR_CUH
#define ONEFLOW_CORE_EP_CUDA_PRIMITIVE_UNARY_FUNCTOR_CUH
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>
#include "oneflow/core/common/math_util.h"

namespace oneflow {
namespace ep {
namespace primitive {

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kGelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Src>(0.5) * src
           * (static_cast<Src>(1.0) + erf(static_cast<Src>(M_SQRT1_2) * src));
  }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kFastGelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    // ref to: https://mlfromscratch.com/activation-functions-explained/#gelu
    const Src half = static_cast<Src>(0.5);
    const Src one = static_cast<Src>(1);
    const Src tanh_in = alpha * (src + beta * src * src * src);
    return half * src * (one + tanh(tanh_in));
  }

 private:
  // constant ref to:
  // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/transform/fusion/fast_gelu.py
  static constexpr Src alpha = static_cast<Src>(0.7978845608028654);
  static constexpr Src beta = static_cast<Src>(0.044714998453855515);
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kQuickGelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    const Src sigmoid =
        static_cast<Dst>(static_cast<Src>(1.0) / (static_cast<Src>(1.0) + exp(-src * alpha)));
    return src * sigmoid;
  }

 private:
  static constexpr Src alpha = static_cast<Src>(1.702);
};

namespace unary_functor_internal {

namespace {

OF_DEVICE_FUNC
float TanhApprox(float x) {
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  return tanhf(x);
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
}

}  // namespace

}  // namespace unary_functor_internal

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kFastGelu, half, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {}

  OF_DEVICE_FUNC half operator()(half src) const {
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
    const float tanh_in =
        __half2float(__float2half_rn(alpha) * (src + __float2half_rn(beta) * src * src * src));
    const float tanh_out = unary_functor_internal::TanhApprox(tanh_in);
    return __float2half_rn(0.5F) * src * (__float2half_rn(1.0F) + __float2half_rn(tanh_out));
#else
    return static_cast<half>(float_functor(static_cast<float>(src)));
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  }

#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  __device__ void Apply2(half* dst, const half* src) const {
    const half2 src2 = *(reinterpret_cast<const half2*>(src));
    const float2 tanh_in = __half22float2(__hmul2(
        __float2half2_rn(alpha),
        __hadd2(src2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), src2), src2), src2))));
    float2 tanh_out;
    tanh_out.x = unary_functor_internal::TanhApprox(tanh_in.x);
    tanh_out.y = unary_functor_internal::TanhApprox(tanh_in.y);
    const half2 dst2 = __hmul2(__hmul2(__float2half2_rn(0.5F), src2),
                               __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));
    *reinterpret_cast<half2*>(dst) = dst2;
  }
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)

 private:
  static constexpr float alpha = 0.7978845608028654F;
  static constexpr float beta = 0.044714998453855515F;
  UnaryFunctor<DeviceType::kCUDA, UnaryOp::kFastGelu, float, float> float_functor;
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, float, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src) const { return tanhf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, double, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src) const { return tanh(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, half, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC half operator()(half src) const { return __float2half(tanhf(__half2float(src))); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(half src) const { return isinf(__half2float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(half src) const { return isnan(__half2float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return isnan(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return isnan(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsFinite, bool, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(half src) const { return isfinite(__half2float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsFinite, bool, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return isfinite(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsFinite, bool, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return isfinite(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTrunc, half, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}
  __device__ half operator()(half src) const { return htrunc(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTrunc, float, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC float operator()(float src) const { return truncf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTrunc, double, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC double operator()(double src) const { return trunc(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kDigamma, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src in) const {
    // references
    // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/cuda/Math.cuh#L3029-L3090
    static const double PI_f64 = 3.14159265358979323846;
    const Src PSI_10 = 2.25175258906672110764;
    const Src A[] = {
        8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
        -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
        8.33333333333333333333E-2,
    };

    Src x = static_cast<Src>(in);
    if (x == static_cast<Src>(0)) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is ±0, ±∞ is returned
      return std::copysign(static_cast<Src>(INFINITY), -x);
    }

    bool x_is_integer = x == trunc(x);
    Src result = static_cast<Src>(0);
    if (x < 0) {
      if (x_is_integer) {
        // As per C++ standard for gamma related functions and SciPy,
        // If the argument is a negative integer, NaN is returned
        return static_cast<Src>(NAN);
      }
      // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
      // accurate than tan(pi * x). While these operations are mathematically equivalent
      // since both x and r are in radians and tan() has a periodicity of pi, in practice
      // the computation of pi * x is a source of error (when |x| > 1).
      double q, r;
      r = modf(static_cast<double>(x), &q);
      result = static_cast<Src>(-PI_f64 / tan(PI_f64 * r));
      x = static_cast<Src>(1) - x;
    }

    while (x < 10) {
      result -= static_cast<Src>(1) / x;
      x += 1;
    }
    if (x == static_cast<Src>(10)) { return static_cast<Src>(result + PSI_10); }

    Src y = 0;
    if (x < 1.0e17) {
      Src z = static_cast<Src>(1) / (x * x);

      Src polevl_result = 0;
      for (int i = 0; i <= 6; i++) { polevl_result = polevl_result * z + A[i]; }
      y = z * polevl_result;
    }

    return static_cast<Src>(log(x) - (static_cast<Src>(0.5) / x) - y + result);
  }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTrigamma, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src x) const {
    // references
    // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/cuda/Math.cuh#L387-L410
    const Src PI{3.14159265358979323846};
    Src sign = 1;
    Src result = 0;

    if (x < Src{0.5}) {
      sign = -1;
      Src sin_pi_x = sin(PI * x);
      result -= (PI * PI) / (sin_pi_x * sin_pi_x);
      x = 1 - x;
    }

    for (int i = 0; i < 6; ++i) {
      result += Src{1} / (x * x);
      x += 1;
    }

    const Src one{1};
    const Src ixx = one / (x * x);
    result += (one + one / (Src{2} * x)
               + ixx * (one / Src{6} - ixx * (one / Src{30} - ixx * (one / Src{42}))))
              / x;
    return sign * result;
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAbs, half, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  __device__ half operator()(half src) const {
    return __hlt(src, static_cast<half>(0)) ? __hneg(src) : src;
  }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kNanAssign, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return isnan(src) ? static_cast<Dst>(0.0) : src; }
};

#if CUDA_VERSION >= 11000
template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAbs, nv_bfloat16, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  __device__ nv_bfloat16 operator()(nv_bfloat16 src) const {
#if CUDA_ARCH >= 800
    return __habs(src);
#else
    return __float2bfloat16(abs(__bfloat162float(src)));
#endif  // CUDA_ARCH >= 800
  }
};
#endif  // CUDA_VERSION >= 11000

/*********half dtype support*********/
template<typename Dst>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, Dst, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(half src) const { return static_cast<Dst>(__half2float(src)); }
};

template<typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, half, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC half operator()(Src src) const { return __float2half(static_cast<float>(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, half, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC half operator()(half src) const { return src; }
};

/*********nv_bfloat16 dtype support*********/
#if CUDA_VERSION >= 11000
template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, nv_bfloat16, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC nv_bfloat16 operator()(half src) const {
    return __float2bfloat16(__half2float(src));
  }
};

template<typename Dst>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, Dst, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(nv_bfloat16 src) const {
    return static_cast<Dst>(__bfloat162float(src));
  }
};

template<typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, nv_bfloat16, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC nv_bfloat16 operator()(Src src) const {
    return __float2bfloat16(static_cast<float>(src));
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, half, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC half operator()(nv_bfloat16 src) const {
    return __float2half(__bfloat162float(src));
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, nv_bfloat16, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC nv_bfloat16 operator()(nv_bfloat16 src) const { return src; }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuComplex, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(nv_bfloat16 src) const {
    return make_cuComplex((__bfloat162float(src)), 0.0);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuDoubleComplex, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuDoubleComplex operator()(nv_bfloat16 src) const {
    return make_cuDoubleComplex(static_cast<double>(__bfloat162float(src)), 0.0);
  }
};

#endif  // CUDA_VERSION >= 11000

#define SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(op)                                         \
  template<>                                                                                 \
  struct UnaryFunctor<DeviceType::kCUDA, op, half, half> {                                   \
    OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                                             \
    UnaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;                         \
    OF_DEVICE_FUNC half operator()(half src) const {                                         \
      return __float2half(float_functor(__half2float(src)));                                 \
    }                                                                                        \
  };

SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kElu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCelu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kGelu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kMish);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSelu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSilu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSoftSign);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSoftPlus);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAcos);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAcosh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAsin);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAsinh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAtan);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAtanh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCeil);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCos);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCosh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kDigamma);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kTrigamma);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kErf);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kErfc);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kExp);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kExp2);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kExpm1);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kFloor);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLgamma);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLog);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLog2);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLog10);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLog1p);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLogSigmoid);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kRint);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kRound);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kRsqrt);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSigmoid);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSin);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSinh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSqrt);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSquare);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kTan);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kReciprocalNoNan);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kNotEqualZero);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kNanAssign);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kQuickGelu);

/*********nv_bfloat16_kernel*******/

#if CUDA_VERSION >= 11000

#define SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(op)                                     \
  template<>                                                                                 \
  struct UnaryFunctor<DeviceType::kCUDA, op, nv_bfloat16, nv_bfloat16> {                     \
    OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                                             \
    UnaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;                         \
    OF_DEVICE_FUNC nv_bfloat16 operator()(nv_bfloat16 src) const {                           \
      return __float2bfloat16(float_functor(__bfloat162float(src)));                         \
    }                                                                                        \
  };

SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kElu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kGelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardSwish);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardSigmoid);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardShrink);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardTanh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLeakyRelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kMish);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSilu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftShrink);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftSign);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftPlus);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTanh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kThreshold);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAcos);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAcosh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAsin);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAsinh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAtan);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAtanh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCeil);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCos);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCosh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kErf);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kErfc);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExp);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExp2);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExpm1);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kFloor);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLgamma);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog2);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog10);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog1p);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLogSigmoid);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRint);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRound);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRsqrt);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSigmoid);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSin);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSinh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSqrt);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSquare);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTan);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kReciprocalNoNan);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kNotEqualZero);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kNanAssign);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kFastGelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kQuickGelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kDigamma);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTrigamma);

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(nv_bfloat16 src) const { return isinf(__bfloat162float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(nv_bfloat16 src) const { return isnan(__bfloat162float(src)); }
};
template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsFinite, bool, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(nv_bfloat16 src) const { return isfinite(__bfloat162float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTrunc, nv_bfloat16, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}
  __device__ nv_bfloat16 operator()(nv_bfloat16 src) const {
#if CUDA_ARCH >= 800
    return htrunc(src);
#else
    return __float2bfloat16(truncf(__bfloat162float(src)));
#endif  // CUDA_ARCH >= 800
  }
};

#endif  // CUDA_VERSION >= 11000

/*********float complex dtype support*********/
template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kConj, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return Dst{src.x, -src.y}; }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kReal, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return static_cast<Dst>(src.x); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kImag, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return static_cast<Dst>(src.y); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kRealGrad, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return Dst{src, 0.0}; }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kImagGrad, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return Dst{0.0, src}; }
};

// avoid warning: narrowing conversion
template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kRealGrad, cuComplex, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(double src) const {
    return cuComplex{static_cast<float>(src), 0.0f};
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kImagGrad, cuComplex, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(double src) const {
    return cuComplex{0.0f, static_cast<float>(src)};
  }
};

template<typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuComplex, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(Src src) const {
    return make_cuComplex(static_cast<float>(src), 0.0);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuComplex, cuComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(cuComplex src) const { return src; }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuComplex, cuDoubleComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(cuDoubleComplex src) const {
    return cuComplexDoubleToFloat(src);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuComplex, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(half src) const {
    return make_cuComplex((__half2float(src)), 0.0);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIdentity, cuComplex, cuComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(cuComplex src) const { return src; }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kConj, cuComplex, cuComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC cuComplex operator()(cuComplex src) const { return cuComplex{src.x, -src.y}; }
};

// reference : thrust: `thrust/detail/complex/csqrtf.h:csqrtf`
template<>
struct UnaryFunctor<kCUDA, UnaryOp::kSqrt, cuComplex, cuComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuComplex operator()(cuComplex src) const {
    float a = src.x, b = src.y;
    float t = 0.0f;
    int scale = 1;
    cuComplex result;

    /* We risk spurious overflow for components >= FLT_MAX / (1 + sqrt(2)). */
    const float THRESH = 1.40949553037932e+38f;

    /* Handle special cases. */
    if (src.x == 0.0f && src.y == float()) {
      // return (complex<float>(0, b));
      return (cuComplex{0.0f, b});
    }

    // FLT_MIN*2
    const float low_thresh = 2.35098870164458e-38f;
    scale = 0;

    if (fabsf(a) >= THRESH || fabsf(b) >= THRESH) {
      /* Scale to avoid overflow. */
      a *= 0.25f;
      b *= 0.25f;
      scale = 1;
    } else if (fabsf(a) <= low_thresh && fabsf(b) <= low_thresh) {
      /* Scale to avoid underflow. */
      a *= 4.f;
      b *= 4.f;
      scale = 2;
    }

    /* Algorithm 312, CACM vol 10, Oct 1967. */
    if (a >= 0.0f) {
      t = sqrtf((a + hypotf(a, b)) * 0.5f);
      // result = complex<float>(t, b / (2.0f * t));
      result.x = t;
      result.y = b / (2.0f * t);
    } else {
      t = sqrtf((-a + hypotf(a, b)) * 0.5f);
      // result = complex<float>(fabsf(b) / (2.0f * t), copysignf(t, b));
      result.x = fabsf(b) / (2.0f * t);
      result.y = copysignf(t, b);
    }

    /* Rescale. */
    if (scale == 1) {
      // return (result * 2.0f);
      result.x *= 2.0f;
      result.y *= 2.0f;
    } else if (scale == 2) {
      // return (result * 0.5f);
      result.x *= 0.5f;
      result.y *= 0.5f;
    }

    return (result);
  }
};

/*********double complex dtype support*********/
template<typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuDoubleComplex, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuDoubleComplex operator()(Src src) const {
    return make_cuDoubleComplex(static_cast<double>(src), 0.0);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuDoubleComplex, cuDoubleComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuDoubleComplex operator()(cuDoubleComplex src) const { return src; }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuDoubleComplex, cuComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuDoubleComplex operator()(cuComplex src) const {
    return cuComplexFloatToDouble(src);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCast, cuDoubleComplex, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuDoubleComplex operator()(half src) const {
    return make_cuDoubleComplex(static_cast<double>(__half2float(src)), 0.0);
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIdentity, cuDoubleComplex, cuDoubleComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuDoubleComplex operator()(cuDoubleComplex src) const { return src; }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kConj, cuDoubleComplex, cuDoubleComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}
  OF_DEVICE_FUNC cuDoubleComplex operator()(cuDoubleComplex src) const {
    return cuDoubleComplex{src.x, -src.y};
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kSqrt, cuDoubleComplex, cuDoubleComplex> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC cuDoubleComplex operator()(cuDoubleComplex src) const {
    double a = src.x, b = src.y;
    double t = 0.0;
    int scale = 1;
    cuDoubleComplex result;

    /* We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2)). */
    const float THRESH = 7.446288774449766337959726e+307;

    /* Handle special cases. */
    if (src.x == 0.0 && src.y == double()) {
      // return (complex<float>(0, b));
      return (cuDoubleComplex{0.0, b});
    }

    // DBL_MIN*2
    const double low_thresh = 4.450147717014402766180465e-308;
    scale = 0;

    if (fabs(a) >= THRESH || fabs(b) >= THRESH) {
      /* Scale to avoid overflow. */
      a *= 0.25;
      b *= 0.25;
      scale = 1;
    } else if (fabs(a) <= low_thresh && fabs(b) <= low_thresh) {
      /* Scale to avoid underflow. */
      a *= 4.0;
      b *= 4.0;
      scale = 2;
    }

    /* Algorithm 312, CACM vol 10, Oct 1967. */
    if (a >= 0.0) {
      t = sqrt((a + hypot(a, b)) * 0.5);
      // result = complex<float>(t, b / (2.0f * t));
      result.x = t;
      result.y = b / (2 * t);
    } else {
      t = sqrt((-a + hypot(a, b)) * 0.5);
      // result = complex<float>(fabsf(b) / (2.0f * t), copysignf(t, b));
      result.x = fabs(b) / (2 * t);
      result.y = copysignf(t, b);
    }

    /* Rescale. */
    if (scale == 1) {
      // return (result * 2.0f);
      result.x *= 2.0;
      result.y *= 2.0;
    } else if (scale == 2) {
      // return (result * 0.5f);
      result.x *= 0.5;
      result.y *= 0.5;
    }

    return (result);
  }
};

#define SPECIALIZATION_COMPLEX_ARITHMETIC_UNARY_FUNCTOR(op, complex_type, real_type)        \
  template<>                                                                                \
  struct UnaryFunctor<DeviceType::kCUDA, op, complex_type, complex_type> {                  \
    OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) : real_functor(attr0, attr1) {} \
    UnaryFunctor<DeviceType::kCUDA, op, real_type, real_type> real_functor;                 \
    OF_DEVICE_FUNC complex_type operator()(complex_type src) const {                        \
      return complex_type{real_functor(src.x), real_functor(src.y)};                        \
    }                                                                                       \
  };

SPECIALIZATION_COMPLEX_ARITHMETIC_UNARY_FUNCTOR(UnaryOp::kNegative, cuComplex, float);
SPECIALIZATION_COMPLEX_ARITHMETIC_UNARY_FUNCTOR(UnaryOp::kNegative, cuDoubleComplex, double);

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
#endif
