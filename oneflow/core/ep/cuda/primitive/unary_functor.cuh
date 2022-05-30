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

namespace oneflow {
namespace ep {
namespace primitive {

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kGelu, Dst, Src> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Src>(0.5) * src
           * (static_cast<Src>(1.0) + erf(static_cast<Src>(M_SQRT1_2) * src));
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, float, float> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src) const { return tanhf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, double, double> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src) const { return tanh(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, half, half> {
  UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC half operator()(half src) const { return __float2half(tanhf(__half2float(src))); }
};

#define SPECIALIZATION_PSEUDO_HALF_FUNCTOR(op)                                \
  template<>                                                                  \
  struct UnaryFunctor<DeviceType::kCUDA, op, half, half> {                    \
    UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                              \
    UnaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;          \
    OF_DEVICE_FUNC half operator()(half src) const {                          \
      return __float2half(float_functor(__half2float(src)));                  \
    }                                                                         \
  };

SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kElu);
SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kCelu);
SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kGelu);
SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kMish);
SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kSelu);
SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kSilu);
SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kSoftSign);
SPECIALIZATION_PSEUDO_HALF_FUNCTOR(UnaryOp::kSoftPlus);

/*********nv_bfloat16_kernel*******/

#if CUDA_VERSION >= 11000

#define SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(op)                            \
  template<>                                                                  \
  struct UnaryFunctor<DeviceType::kCUDA, op, nv_bfloat16, nv_bfloat16> {      \
    UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                              \
    UnaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;          \
    OF_DEVICE_FUNC nv_bfloat16 operator()(nv_bfloat16 src) const {            \
      return __float2bfloat16(float_functor(__bfloat162float(src)));          \
    }                                                                         \
  };

SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kElu);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kCelu);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kGelu);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kHardSwish);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kHardSigmoid);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kHardShrink);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kHardTanh);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kLeakyRelu);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kMish);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kSelu);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kSilu);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kSoftShrink);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kSoftSign);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kSoftPlus);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kTanh);
SPECIALIZATION_PSEUDO_BFLOAT16_FUNCTOR(UnaryOp::kThreshold);

#endif

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
