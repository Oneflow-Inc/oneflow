#include "oneflow/customized/kernels/scalar_binary_kernel.h"

namespace oneflow {

namespace {

template<template<typename> class binary_func, typename T, typename Index>
struct LeftScalarBinaryCalculation<binary_func, DeviceType::kCPU, T, Index> {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const Index n) {
    for (Index i = 0; i < n; ++i) { y_ptr[i] = binary_func<T>::Invoke(scalar_operand, x_ptr[i]); }
  }
};

template<template<typename> class binary_func, typename T, typename Index>
struct RightScalarBinaryCalculation<binary_func, DeviceType::kCPU, T, Index> {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const Index n) {
    for (Index i = 0; i < n; ++i) { y_ptr[i] = binary_func<T>::Invoke(x_ptr[i], scalar_operand); }
  }
};
}  // namespace

#define ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  FLOATING_DATA_TYPE_SEQ

#define REGISTER_ALL_CPU_SCALAR_BINARY_KERNELS(dtype, _) \
  REGISTER_ALL_SCALAR_BINARY_KERNELS(CPU, dtype)

OF_PP_FOR_EACH_TUPLE(REGISTER_ALL_CPU_SCALAR_BINARY_KERNELS, ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8)
REGISTER_SCALAR_BINARY_KERNEL("scalar_add", Commutative, BinaryFuncAdd, CPU, int8_t)

}  // namespace oneflow
