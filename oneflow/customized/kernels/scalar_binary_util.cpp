#include "oneflow/customized/kernels/scalar_binary_util.h"

namespace oneflow {

template<template<typename> class binary_func, typename T>
struct ScalarBinary<binary_func, BinaryOpType::kRight, DeviceType::kCPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* in_ptr, const T scalar_operand, T* out_ptr,
                     const int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = binary_func<T>::Invoke(in_ptr[i], scalar_operand);
    }
  }
  static void InvokeInplace(DeviceCtx* ctx, T* in_ptr, const T scalar_operand, const int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      in_ptr[i] = binary_func<T>::Invoke(in_ptr[i], scalar_operand);
    }
  }
};

template<template<typename> class binary_func, BinaryOpType binary_op_type, typename T>
struct ScalarBinary<binary_func, binary_op_type, DeviceType::kCPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* in_ptr, const T scalar_operand, T* out_ptr,
                     const int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = binary_func<T>::Invoke(scalar_operand, in_ptr[i]);
    }
  }
  static void InvokeInplace(DeviceCtx* ctx, T* in_ptr, const T scalar_operand, const int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      in_ptr[i] = binary_func<T>::Invoke(scalar_operand, in_ptr[i]);
    }
  }
};

#define INSTANTIATE_KERNEL_WITH_TYPE(type, _)                                                \
  template struct ScalarBinary<BinaryFuncAdd, BinaryOpType::kCommutative, DeviceType::kCPU, type>;\
  template struct ScalarBinary<BinaryFuncMul, BinaryOpType::kCommutative, DeviceType::kCPU, type>;\
  template struct ScalarBinary<BinaryFuncDiv, BinaryOpType::kLeft, DeviceType::kCPU, type>;\
  template struct ScalarBinary<BinaryFuncDiv, BinaryOpType::kRight, DeviceType::kCPU, type>;\

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

}
