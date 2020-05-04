#include "oneflow/customized/kernels/scalar_binary_kernel.h"

namespace oneflow {

template<template<typename> class binary_func, typename T>
class LeftBinaryKernel<binary_func, DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  LeftBinaryKernel() = default;
  ~LeftBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* x_ptr = GetXPtr<T>(ctx);
    auto* y_ptr = GetYPtr<T>(ctx);
    const auto scalar_operand = GetScalarOperand<T>(ctx);
    const auto n = GetElemCnt(ctx);

    if (y_ptr == x_ptr) {
      for (int64_t i = 0; i < n; ++i) {
        y_ptr[i] = binary_func<T>::Invoke(scalar_operand, y_ptr[i]);
      }
    } else {
      for (int64_t i = 0; i < n; ++i) {
        y_ptr[i] = binary_func<T>::Invoke(scalar_operand, x_ptr[i]);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class binary_func, typename T>
class RightBinaryKernel<binary_func, DeviceType::kCPU, T> final : public user_op::OpKernel {
 public:
  RightBinaryKernel() = default;
  ~RightBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* x_ptr = GetXPtr<T>(ctx);
    auto* y_ptr = GetYPtr<T>(ctx);
    const auto scalar_operand = GetScalarOperand<T>(ctx);
    const auto n = GetElemCnt(ctx);

    if (y_ptr == x_ptr) {
      for (int64_t i = 0; i < n; ++i) {
        y_ptr[i] = binary_func<T>::Invoke(y_ptr[i], scalar_operand);
      }
    } else {
      for (int64_t i = 0; i < n; ++i) {
        y_ptr[i] = binary_func<T>::Invoke(x_ptr[i], scalar_operand);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(op_name, kernel_type, func_name, kernel_device_type, dtype)         \
  REGISTER_USER_KERNEL(op_name)                                                             \
      .SetCreateFn<                                                                         \
          kernel_type##BinaryKernel<func_name, DeviceType::k##kernel_device_type, dtype>>() \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                          \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);       \
        return ctx.device_type() == DeviceType::k##kernel_device_type                       \
               && y_desc->data_type() == GetDataType<dtype>::value;                         \
      });

#define REGISTER_ADD_KERNEL_WITH_TYPE(type, _) \
  REGISTER_KERNEL("scalar_add", Commutative, BinaryFuncAdd, CPU, type)

OF_PP_FOR_EACH_TUPLE(REGISTER_ADD_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

#undef REGISTER_ADD_KERNEL_WITH_TYPE

#define ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  FLOATING_DATA_TYPE_SEQ

#define REGISTER_MUL_DIV_KERNEL_WITH_TYPE(type, _)                     \
  REGISTER_KERNEL("scalar_mul", Commutative, BinaryFuncMul, CPU, type) \
  REGISTER_KERNEL("left_scalar_div", Left, BinaryFuncDiv, CPU, type)   \
  REGISTER_KERNEL("right_scalar_div", Right, BinaryFuncDiv, CPU, type)

// OF_PP_FOR_EACH_TUPLE(REGISTER_MUL_DIV_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8)

#undef REGISTER_MUL_DIV_KERNEL_WITH_TYPE
#undef ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8

}  // namespace oneflow
