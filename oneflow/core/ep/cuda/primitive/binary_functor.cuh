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

#include "oneflow/core/ep/common/primitive/binary_functor.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kPow, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return pow(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kPow, bool, bool> {
  OF_DEVICE_FUNC bool operator()(bool src0, bool src1) const {
    return static_cast<bool>(pow(static_cast<double>(src0), static_cast<double>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kPow, half, half> {
  OF_DEVICE_FUNC half operator()(half src0, half src1) const {
    return static_cast<half>(pow(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

#if CUDA_VERSION >= 11000

template<>
struct BinaryFunctor<DeviceType::kCUDA, BinaryOp::kPow, nv_bfloat16, nv_bfloat16> {
  OF_DEVICE_FUNC nv_bfloat16 operator()(nv_bfloat16 src0, nv_bfloat16 src1) const {
    return static_cast<nv_bfloat16>(pow(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

#endif  // CUDA_VERSION >= 11000

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
