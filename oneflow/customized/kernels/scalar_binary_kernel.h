#ifndef ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_KERNEL_H
#define ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_KERNEL_H

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T>
const T* GetInPtr(user_op::KernelComputeContext* ctx) {
  const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
  const T* in_ptr = in->dptr<T>();
  return in_ptr;
}

template<typename T>
T* GetOutPtr(user_op::KernelComputeContext* ctx) {
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  T* out_ptr = out->mut_dptr<T>();
  return out_ptr;
}

template<typename T>
const T GetScalarOperand(user_op::KernelComputeContext* ctx) {
  T scalar_operand = static_cast<T>(0);
  if (ctx->GetAttr<bool>("has_int_operand")) {
    scalar_operand = static_cast<T>(ctx->GetAttr<int64_t>("int_operand"));
  } else if (ctx->GetAttr<bool>("has_float_operand")) {
    scalar_operand = static_cast<T>(ctx->GetAttr<double>("float_operand"));
  } else {
    UNIMPLEMENTED();
  }
  return scalar_operand;
}

inline const int64_t GetElemCnt(user_op::KernelComputeContext* ctx) {
  const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
  return in->shape().elem_cnt();
}

template<template<typename> class binary_func, DeviceType device_type, typename T>
class LeftBinaryKernel;

template<template<typename> class binary_func, DeviceType device_type, typename T>
using CommutativeBinaryKernel = LeftBinaryKernel<binary_func, device_type, T>;

template<template<typename> class binary_func, DeviceType device_type, typename T>
class RightBinaryKernel;

}  // namespace oneflow

#endif /* ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_KERNEL_H */
