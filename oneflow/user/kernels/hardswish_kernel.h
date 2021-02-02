#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_HARDSWISH_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_HARDSWISH_KERNEL_H_
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

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

#define REGISTER_HARDSWISH_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("hardswish")                                                \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                              \
        return new UnaryElemwiseXpuKernel<device, HardswishFunctor<dtype>, dtype>(            \
            [](user_op::KernelComputeContext* ctx) {                                    \
              return HardswishFunctor<dtype>();                     \
            },                                                                          \
            "out", "in");                                                               \
      })                                                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("hardswish_grad")                                                      \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                              \
        return new BinaryElemwiseXpuKernel<device, HardswishGradFunctor<dtype>, dtype>(       \
            [](user_op::KernelComputeContext* ctx) {                                    \
              return HardswishGradFunctor<dtype>();                 \
            },                                                                          \
            "dx", "x", "dy");                                                           \
      })                                                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

} // namespace oneflow

#endif 