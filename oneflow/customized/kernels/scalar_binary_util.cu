#include "oneflow/customized/kernels/scalar_binary_util.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

template<template<typename> class binary_func, typename T>
__global__ void ScalarBinaryRightGpu(const T* in_ptr, const T scalar_operand, T* out_ptr,
                                     const int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { out_ptr[i] = binary_func<T>::Invoke(in_ptr[i], scalar_operand); }
}
template<template<typename> class binary_func, typename T>
__global__ void ScalarBinaryRightGpuInplace(T* in_ptr, const T scalar_operand, const int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { in_ptr[i] = binary_func<T>::Invoke(in_ptr[i], scalar_operand); }
}
template<template<typename> class binary_func, typename T>
__global__ void ScalarBinaryLeftGpu(const T* in_ptr, const T scalar_operand, T* out_ptr,
                                    const int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { out_ptr[i] = binary_func<T>::Invoke(scalar_operand, in_ptr[i]); }
}
template<template<typename> class binary_func, typename T>
__global__ void ScalarBinaryLeftGpuInplace(T* in_ptr, const T scalar_operand, const int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { in_ptr[i] = binary_func<T>::Invoke(scalar_operand, in_ptr[i]); }
}
}  // namespace

template<template<typename> class binary_func, BinaryOpType binary_op_type, typename T>
struct ScalarBinary<binary_func, binary_op_type, DeviceType::kGPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* in_ptr, const T scalar_operand, T* out_ptr,
                     const int64_t n) {
    RUN_CUDA_KERNEL((ScalarBinaryLeftGpu<binary_func, T>), ctx, n, in_ptr, scalar_operand, out_ptr,
                    n);
  }
  static void InvokeInplace(DeviceCtx* ctx, T* in_ptr, const T scalar_operand, const int64_t n) {
    RUN_CUDA_KERNEL((ScalarBinaryLeftGpuInplace<binary_func, T>), ctx, n, in_ptr, scalar_operand,
                    n);
  }
};

template<template<typename> class binary_func, typename T>
struct ScalarBinary<binary_func, BinaryOpType::kRight, DeviceType::kGPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* in_ptr, const T scalar_operand, T* out_ptr,
                     const int64_t n) {
    RUN_CUDA_KERNEL((ScalarBinaryRightGpu<binary_func, T>), ctx, n, in_ptr, scalar_operand, out_ptr,
                    n);
  }
  static void InvokeInplace(DeviceCtx* ctx, T* in_ptr, const T scalar_operand, const int64_t n) {
    RUN_CUDA_KERNEL((ScalarBinaryRightGpuInplace<binary_func, T>), ctx, n, in_ptr, scalar_operand,
                    n);
  }
};

#define INSTANTIATE_KERNEL_WITH_TYPE(type, _)                                                \
  template struct ScalarBinary<BinaryFuncAdd, BinaryOpType::kCommutative, DeviceType::kGPU, type>;\
  template struct ScalarBinary<BinaryFuncMul, BinaryOpType::kCommutative, DeviceType::kGPU, type>;\
  template struct ScalarBinary<BinaryFuncDiv, BinaryOpType::kLeft, DeviceType::kGPU, type>;\
  template struct ScalarBinary<BinaryFuncDiv, BinaryOpType::kRight, DeviceType::kGPU, type>;\

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
