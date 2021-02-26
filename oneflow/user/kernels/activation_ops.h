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
#ifndef _ONEFLOW_USER_KERNELS_ACTIVATION_OPS_H_
#define _ONEFLOW_USER_KERNELS_ACTIVATION_OPS_H_
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

template<typename T>
struct EluFunctor {
  OF_DEVICE_FUNC explicit EluFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x > static_cast<T>(0)) ? x : static_cast<T>(alpha * (exp(x) - static_cast<T>(1)));
  }
  const T alpha;
};

template<typename T>
struct EluGradFunctor {
  OF_DEVICE_FUNC explicit EluGradFunctor(float alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x, T dy) const {
    return (x > static_cast<T>(0)) ? dy : static_cast<T>(dy * alpha * (exp(x)));
  }
  const T alpha;
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

template<typename T>
struct HardtanhFunctor {
  OF_DEVICE_FUNC explicit HardtanhFunctor(float min_val, float max_val)
      : min_val(min_val), max_val(max_val) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    if (x <= min_val) {
      return min_val;
    } else if (x >= max_val) {
      return max_val;
    } else {
      return x;
    }
  }
  const T min_val;
  const T max_val;
};

template<typename T>
struct HardtanhGradFunctor {
  OF_DEVICE_FUNC explicit HardtanhGradFunctor(float min_val, float max_val)
      : min_val(min_val), max_val(max_val) {}
  OF_DEVICE_FUNC T operator()(T y, T dy) const {
    return (y != min_val && y != max_val) ? dy : static_cast<T>(0);
  }
  const T min_val;
  const T max_val;
};

#define REGISTER_ELU_KERNEL(device, dtype)                                                         \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL_WITH_ATTR(device, "elu", EluFunctor, dtype, dtype, "out",    \
                                                "in", ctx->Attr<double>("alpha"));                 \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL_WITH_ATTR(device, "elu_grad", EluGradFunctor, dtype, dtype, \
                                                 dtype, "dx", "x", "dy",                           \
                                                 ctx->Attr<double>("alpha"));

#define REGISTER_HARDSWISH_KERNEL(device, dtype)                                                 \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL_WITHOUT_ATTR(device, "hardswish", HardswishFunctor, dtype, \
                                                   dtype, "out", "in");                          \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL_WITHOUT_ATTR(                                             \
      device, "hardswish_grad", HardswishGradFunctor, dtype, dtype, dtype, "dx", "x", "dy");

#define REGISTER_HARDSIGMOID_KERNEL(device, dtype)                                            \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL_WITHOUT_ATTR(device, "hardsigmoid", HardsigmoidFunctor, \
                                                   dtype, dtype, "out", "in");                \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL_WITHOUT_ATTR(                                          \
      device, "hardsigmoid_grad", HardsigmoidGradFunctor, dtype, dtype, dtype, "dx", "x", "dy");

#define REGISTER_HARDTANH_KERNEL(device, dtype)                                                    \
  REGISTER_UNARY_ELEMWISE_USER_KERNEL_WITH_ATTR(device, "hardtanh", HardtanhFunctor, dtype, dtype, \
                                                "out", "in", ctx->Attr<double>("min_val"),         \
                                                ctx->Attr<double>("max_val"));

// #define REGISTER_HARDTANH_KERNEL(device, dtype)                                                 \
//   REGISTER_USER_KERNEL("hardtanh")                                                              \
//       .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                      \
//         return new UnaryElemwiseXpuKernel<device, HardtanhFunctor<dtype>, dtype, dtype>(        \
//             [](user_op::KernelComputeContext* ctx) {                                            \
//               return HardtanhFunctor<dtype>(ctx->Attr<double>("min_val"),                       \
//                                             ctx->Attr<double>("max_val"));                      \
//             },                                                                                  \
//             "out", "in");                                                                       \
//       })                                                                                        \
//       .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
//                        & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))          \
//       .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
//                                user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
//         OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
//         return Maybe<void>::Ok();                                                               \
//       });                                                                                       \
//   REGISTER_USER_KERNEL("hardtanh_grad")                                                         \
//       .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                      \
//         return new BinaryElemwiseXpuKernel<device, HardtanhGradFunctor<dtype>, dtype, dtype,    \
//                                            dtype>(                                              \
//             [](user_op::KernelComputeContext* ctx) {                                            \
//               return HardtanhGradFunctor<dtype>(ctx->Attr<double>("min_val"),                   \
//                                                 ctx->Attr<double>("max_val"));                  \
//             },                                                                                  \
//             "dx", "y", "dy");                                                                   \
//       })                                                                                        \
//       .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
//                        & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
//       .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
//                                user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
//         OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
//         return Maybe<void>::Ok();                                                               \
//       });

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ACTIVATION_OPS_H_
