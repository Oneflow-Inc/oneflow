#ifndef ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_KERNEL_H
#define ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_KERNEL_H

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T>
const T* GetXPtr(user_op::KernelComputeContext* ctx) {
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  const T* x_ptr = x->dptr<T>();
  return x_ptr;
}

template<typename T>
T* GetYPtr(user_op::KernelComputeContext* ctx) {
  user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
  T* y_ptr = y->mut_dptr<T>();
  return y_ptr;
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
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  return x->shape().elem_cnt();
}

template<template<typename> class binary_func, DeviceType device_type, typename T>
struct LeftScalarBinaryCalculation {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const int64_t n);
};

template<template<typename> class binary_func, DeviceType device_type, typename T>
using CommutativeScalarBinaryCalculation = LeftScalarBinaryCalculation<binary_func, device_type, T>;

template<template<typename> class binary_func, DeviceType device_type, typename T>
struct RightScalarBinaryCalculation {
  static void Invoke(DeviceCtx* ctx, const T* x_ptr, const T scalar_operand, T* y_ptr,
                     const int64_t n);
};

template<template<typename> class binary_func, DeviceType device_type, typename T>
class LeftScalarBinaryKernel final : public user_op::OpKernel {
 public:
  LeftScalarBinaryKernel() = default;
  ~LeftScalarBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LeftScalarBinaryCalculation<binary_func, device_type, T>::Invoke(
        ctx->device_ctx(), GetXPtr<T>(ctx), GetScalarOperand<T>(ctx), GetYPtr<T>(ctx),
        GetElemCnt(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class binary_func, DeviceType device_type, typename T>
using CommutativeScalarBinaryKernel = LeftScalarBinaryKernel<binary_func, device_type, T>;

template<template<typename> class binary_func, DeviceType device_type, typename T>
class RightScalarBinaryKernel final : public user_op::OpKernel {
 public:
  RightScalarBinaryKernel() = default;
  ~RightScalarBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    RightScalarBinaryCalculation<binary_func, device_type, T>::Invoke(
        ctx->device_ctx(), GetXPtr<T>(ctx), GetScalarOperand<T>(ctx), GetYPtr<T>(ctx),
        GetElemCnt(ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALAR_BINARY_KERNEL(op_name, kernel_type, func_name, kernel_device_type, dtype) \
  REGISTER_USER_KERNEL(op_name)                                                                   \
      .SetCreateFn<                                                                               \
          kernel_type##ScalarBinaryKernel<func_name, DeviceType::k##kernel_device_type, dtype>>() \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);               \
        return ctx.device_type() == DeviceType::k##kernel_device_type                             \
               && y_desc->data_type() == GetDataType<dtype>::value;                               \
      })                                                                                          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                      \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {   \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                            \
        return Maybe<void>::Ok();                                                                 \
      });

#define REGISTER_ALL_SCALAR_BINARY_KERNELS(device_type, dtype)                                \
  REGISTER_SCALAR_BINARY_KERNEL("scalar_add", Commutative, BinaryFuncAdd, device_type, dtype) \
  REGISTER_SCALAR_BINARY_KERNEL("scalar_mul", Commutative, BinaryFuncMul, device_type, dtype) \
  REGISTER_SCALAR_BINARY_KERNEL("left_scalar_div", Left, BinaryFuncDiv, device_type, dtype)   \
  REGISTER_SCALAR_BINARY_KERNEL("right_scalar_div", Right, BinaryFuncDiv, device_type, dtype)

}  // namespace oneflow

#endif /* ONEFLOW_CUSTOMIZED_KERNELS_SCALAR_BINARY_KERNEL_H */
