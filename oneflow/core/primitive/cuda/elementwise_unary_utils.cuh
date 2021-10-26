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
#include "oneflow/core/primitive/common/elementwise_unary_utils.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {
namespace primitive {

template<>
struct UnaryFunctor<DeviceType::kGPU, UnaryOp::kRelu, half> {
  __device__ half operator()(half src) const {
    half zero_half = static_cast<half>(0.0);
    if (__hlt(src, zero_half)) {
      return zero_half;
    } else {
      return src;
    }
  }
};

#if CUDA_VERSION >= 11000

template<>
struct UnaryFunctor<DeviceType::kGPU, UnaryOp::kRelu, nv_bfloat16> {
  __device__ nv_bfloat16 operator()(nv_bfloat16 src) const {
    const nv_bfloat16 zero_bfloat16 = static_cast<nv_bfloat16>(0.0);
    if (src > zero_bfloat16) {
      return src;
    } else {
      return zero_bfloat16;
    }
  }
};
#endif  // CUDA_VERSION >= 11000
}  // namespace primitive
}  // namespace oneflow
