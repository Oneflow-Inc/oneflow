#include "oneflow/customized/kernels/scalar_binary_kernel.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace {

template<template<typename> class binary_func, typename T>
__global__ void RightScalarBinaryGpu(const T* x_ptr, const T scalar_operand, T* y_ptr,
                                     const int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { y_ptr[i] = binary_func<T>::Invoke(x_ptr[i], scalar_operand); }
}
template<template<typename> class binary_func, typename T>
__global__ void LeftScalarBinaryGpu(const T* x_ptr, const T scalar_operand, T* y_ptr,
                                    const int64_t n) {
  CUDA_1D_KERNEL_LOOP(i, n) { y_ptr[i] = binary_func<T>::Invoke(scalar_operand, x_ptr[i]); }
}

template<template<typename> class binary_func, typename T>
struct LeftScalarBinaryCalculation<binary_func, DeviceType::kGPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const int64_t n) {
    RUN_CUDA_KERNEL((LeftScalarBinaryGpu<binary_func, T>), ctx, n, x_ptr, scalar_operand, y_ptr, n);
  }
};

template<template<typename> class binary_func, typename T>
struct RightScalarBinaryCalculation<binary_func, DeviceType::kGPU, T> {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const int64_t n) {
    RUN_CUDA_KERNEL((RightScalarBinaryGpu<binary_func, T>), ctx, n, x_ptr, scalar_operand, y_ptr,
                    n);
  }
};
}  // namespace

#define ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  FLOATING_DATA_TYPE_SEQ

#define REGISTER_ALL_GPU_SCALAR_BINARY_KERNELS(dtype, _) \
  REGISTER_ALL_SCALAR_BINARY_KERNELS(GPU, dtype)

OF_PP_FOR_EACH_TUPLE(REGISTER_ALL_GPU_SCALAR_BINARY_KERNELS, ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8)
REGISTER_SCALAR_BINARY_KERNEL("scalar_add", Commutative, BinaryFuncAdd, GPU, int8_t)

}  // namespace oneflow
