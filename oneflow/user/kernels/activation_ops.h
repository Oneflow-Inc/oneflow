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
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

template<typename T>
struct EluFunctor {
  OF_DEVICE_FUNC explicit EluFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x > static_cast<T>(0)) ? x : static_cast<T>(alpha * (exp(x) - static_cast<T>(1)));
  }
  const float alpha;
};

template<typename T>
struct EluGradFunctor {
  OF_DEVICE_FUNC explicit EluGradFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(0)) ? dy : static_cast<T>(dy * alpha * (exp(x)));
  }
  const float alpha;
};

template<typename T>
struct HardswishFunctor {
  OF_DEVICE_FUNC T operator()(const T x) const {
    if (x <= static_cast<T>(-3)) {
      return static_cast<T>(0);
    } else if (x >= static_cast<T>(3)) {
      return x;
    } else {
      return (x * (x + static_cast<T>(3))) / static_cast<T>(6);
    }
  }
};

template<typename T>
struct HardswishGradFunctor {
  OF_DEVICE_FUNC T operator()(const T x, const T dy) const {
    if (x <= static_cast<T>(-3)) {
      return static_cast<T>(0);
    } else if (x >= static_cast<T>(3)) {
      return dy;
    } else {
      return ((x / static_cast<T>(3)) + static_cast<T>(0.5)) * dy;
    }
  }
};

template<typename T>
struct HardsigmoidFunctor {
  OF_DEVICE_FUNC T operator()(T x) const {
    if (x <= static_cast<T>(-3))
      return static_cast<T>(0);
    else if (x >= static_cast<T>(3))
      return static_cast<T>(1);
    else
      return (x / static_cast<T>(6)) + static_cast<T>(0.5);
  }
};

template<typename T>
struct HardsigmoidGradFunctor {
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(-3) && x < static_cast<T>(3)) ? dy / static_cast<T>(6)
                                                             : static_cast<T>(0);
  }
};

#define REGISTER_ELU_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("elu")                                                           \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                              \
        return new UnaryElemwiseXpuKernel<device, EluFunctor<dtype>, dtype>(            \
            [](user_op::KernelComputeContext* ctx) {                                    \
              return EluFunctor<dtype>(ctx->Attr<float>("alpha"));                      \
            },                                                                          \
            "out", "in");                                                               \
      })                                                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("elu_grad")                                                      \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                              \
        return new BinaryElemwiseXpuKernel<device, EluGradFunctor<dtype>, dtype>(       \
            [](user_op::KernelComputeContext* ctx) {                                    \
              return EluGradFunctor<dtype>(ctx->Attr<float>("alpha"));                  \
            },                                                                          \
            "dx", "x", "dy");                                                           \
      })                                                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_HARDSWISH_KERNEL(device, dtype)                                                 \
  REGISTER_USER_KERNEL("hardswish")                                                              \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                       \
        return new UnaryElemwiseXpuKernel<device, HardswishFunctor<dtype>, dtype>(               \
            [](user_op::KernelComputeContext* ctx) { return HardswishFunctor<dtype>(); }, "out", \
            "in");                                                                               \
      })                                                                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                       \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));          \
  REGISTER_USER_KERNEL("hardswish_grad")                                                         \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                       \
        return new BinaryElemwiseXpuKernel<device, HardswishGradFunctor<dtype>, dtype>(          \
            [](user_op::KernelComputeContext* ctx) { return HardswishGradFunctor<dtype>(); },    \
            "dx", "x", "dy");                                                                    \
      })                                                                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                       \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_HARDSIGMOID_KERNEL(device, dtype)                                                 \
  REGISTER_USER_KERNEL("hardsigmoid")                                                              \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                         \
        return new UnaryElemwiseXpuKernel<device, HardsigmoidFunctor<dtype>, dtype>(               \
            [](user_op::KernelComputeContext* ctx) { return HardsigmoidFunctor<dtype>(); }, "out", \
            "in");                                                                                 \
      })                                                                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                         \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));            \
  REGISTER_USER_KERNEL("hardsigmoid_grad")                                                         \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                         \
        return new BinaryElemwiseXpuKernel<device, HardsigmoidGradFunctor<dtype>, dtype>(          \
            [](user_op::KernelComputeContext* ctx) { return HardsigmoidGradFunctor<dtype>(); },    \
            "dx", "x", "dy");                                                                      \
      })                                                                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                         \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

}  // namespace oneflow
