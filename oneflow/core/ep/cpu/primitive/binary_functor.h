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
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kPow, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return std::pow(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kPow, float16, float16> {
  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::pow(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
