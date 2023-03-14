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
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>

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

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
