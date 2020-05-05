#include "oneflow/customized/kernels/scalar_binary_kernel.h"

namespace oneflow {

template<template<typename> class binary_func, typename T>
struct LeftScalarBinaryCalculation<binary_func, DeviceType::kCPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const int64_t n) {
    for (int64_t i = 0; i < n; ++i) { y_ptr[i] = binary_func<T>::Invoke(scalar_operand, x_ptr[i]); }
  }
};

template<template<typename> class binary_func, typename T>
struct RightScalarBinaryCalculation<binary_func, DeviceType::kCPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const int64_t n) {
    for (int64_t i = 0; i < n; ++i) { y_ptr[i] = binary_func<T>::Invoke(x_ptr[i], scalar_operand); }
  }
};

#define ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  FLOATING_DATA_TYPE_SEQ

#define REGISTER_ALL_CPU_SCALAR_BINARY_KERNELS(dtype, _) \
  REGISTER_ALL_SCALAR_BINARY_KERNELS(CPU, dtype)

OF_PP_FOR_EACH_TUPLE(REGISTER_ALL_CPU_SCALAR_BINARY_KERNELS, ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8)
REGISTER_SCALAR_BINARY_KERNEL("scalar_add", Commutative, BinaryFuncAdd, CPU, int8_t)

#define INSTANTIATE_ALL_CPU_SCALAR_BINARY_CALS(dtype, _) \
  INSTANTIATE_ALL_SCALAR_BINARY_CALS(CPU, dtype)
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_ALL_CPU_SCALAR_BINARY_CALS, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
