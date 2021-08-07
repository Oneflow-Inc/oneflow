#ifndef _ONEFLOW_USER_KERNELS_SCALAR_LOGICAL_KERNELS_H_
#define _ONEFLOW_USER_KERNELS_SCALAR_LOGICAL_KERNELS_H_
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

template<typename T>
struct ScalarLogicalEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const {
    return (x == scalar) ? static_cast<int8_t>(1): static_cast<int8_t>(0);
  }
  const T scalar;
};

template<typename T>
struct ScalarLogicalNotEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalNotEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const {
    return (x != scalar) ? static_cast<int8_t>(1): static_cast<int8_t>(0);
  }
  const T scalar;
};

template<typename T>
struct ScalarLogicalGreaterFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalGreaterFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const {
    return (x > scalar) ? static_cast<int8_t>(1): static_cast<int8_t>(0);
  }
  const T scalar;
};

template<typename T>
struct ScalarLogicalGreaterEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalGreaterEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const {
    return (x >= scalar) ? static_cast<int8_t>(1): static_cast<int8_t>(0);
  }
  const T scalar;
};

template<typename T>
struct ScalarLogicalLessFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalLessFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const {
    return (x < scalar) ? static_cast<int8_t>(1): static_cast<int8_t>(0);
  }
  const T scalar;
};

template<typename T>
struct ScalarLogicalLessEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalLessEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const {
    return (x <= scalar) ? static_cast<int8_t>(1): static_cast<int8_t>(0);
  }
  const T scalar;
};

#define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, kernel_name, functor, out_dtype,             \
                                            input_a_dtype, create_function, out_name,            \
                                            input_a_name)                                        \
  REGISTER_USER_KERNEL(kernel_name)                                                              \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                       \
        return new UnaryElemwiseXpuKernel<device, functor<out_dtype>, out_dtype, input_a_dtype>( \
            create_function, out_name, input_a_name);                                            \
      })                                                                                         \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == device));


#define REGISTER_SCALAR_LOGICAL_EQUAL_KERNEL(device, dtype)                        \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                            \
      device, "scalar_logical_equal", ScalarLogicalEqualFunctor, dtype, dtype,                    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return ScalarLogicalEqualFunctor<dtype>(ctx->Attr<double>("scalar"));     \
      },"out", "in");                                               

#define REGISTER_SCALAR_LOGICAL_NOTEQUAL_KERNEL(device, dtype)                        \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                            \
      device, "scalar_logical_not_equal", ScalarLogicalNotEqualFunctor, dtype, dtype,                    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return ScalarLogicalNotEqualFunctor<dtype>(ctx->Attr<double>("scalar"));     \
      },"out", "in");  

#define REGISTER_SCALAR_LOGICAL_GREATER_KERNEL(device, dtype)                        \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                            \
      device, "scalar_logical_greater", ScalarLogicalGreaterFunctor, dtype, dtype,                    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return ScalarLogicalGreaterFunctor<dtype>(ctx->Attr<double>("scalar"));     \
      },"out", "in");  

#define REGISTER_SCALAR_LOGICAL_GREATER_EQUAL_KERNEL(device, dtype)                        \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                            \
      device, "scalar_logical_greater_equal", ScalarLogicalGreaterEqualFunctor, dtype, dtype,                    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return ScalarLogicalGreaterEqualFunctor<dtype>(ctx->Attr<double>("scalar"));     \
      },"out", "in");  

#define REGISTER_SCALAR_LOGICAL_LESS_KERNEL(device, dtype)                        \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                            \
      device, "scalar_logical_less", ScalarLogicalLessFunctor, dtype, dtype,                    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return ScalarLogicalLessFunctor<dtype>(ctx->Attr<double>("scalar"));     \
      },"out", "in");  

#define REGISTER_SCALAR_LOGICAL_LESS_EQUAL_KERNEL(device, dtype)                        \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                            \
      device, "scalar_logical_less_equal", ScalarLogicalLessEqualFunctor, dtype, dtype,                    \
      [](user_op::KernelComputeContext* ctx) {                    \
        return ScalarLogicalLessEqualFunctor<dtype>(ctx->Attr<double>("scalar"));     \
      },"out", "in");  

}

#endif //_ONEFLOW_USER_KERNELS_SCALAR_LOGICAL_KERNELS_H_
