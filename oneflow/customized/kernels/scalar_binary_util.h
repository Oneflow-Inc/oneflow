#ifndef ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_UTIL_H
#define ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_UTIL_H

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

enum BinaryOpType {
  kCommutative,
  kLeft,
  kRight,
};

template<template<typename> class binary_func, BinaryOpType op_type, DeviceType device_type,
         typename T>
struct ScalarBinary {
  static void Invoke(DeviceCtx* ctx, const T* in_ptr, const T scalar_operand, T* out_ptr,
                     const int64_t n);
  static void InvokeInplace(DeviceCtx* ctx, T* in_ptr, const T scalar_operand, const int64_t n);
};

}  // namespace oneflow

#endif /* ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_UTIL_H */
