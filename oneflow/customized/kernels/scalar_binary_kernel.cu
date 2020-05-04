#include "oneflow/customized/kernels/scalar_binary_kernel.h"
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

template<template<typename> class binary_func, typename T>
class LeftBinaryKernel<binary_func, DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  LeftBinaryKernel() = default;
  ~LeftBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* in_ptr = GetInPtr<T>(ctx);
    auto* out_ptr = GetOutPtr<T>(ctx);
    const auto scalar_operand = GetScalarOperand<T>(ctx);
    const auto n = GetElemCnt(ctx);

    if (out_ptr == in_ptr) {
      RUN_CUDA_KERNEL((ScalarBinaryLeftGpuInplace<binary_func, T>), ctx->device_ctx(), n, out_ptr,
                      scalar_operand, n);
    } else {
      RUN_CUDA_KERNEL((ScalarBinaryLeftGpu<binary_func, T>), ctx->device_ctx(), n, in_ptr,
                      scalar_operand, out_ptr, n);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename> class binary_func, typename T>
class RightBinaryKernel<binary_func, DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  RightBinaryKernel() = default;
  ~RightBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* in_ptr = GetInPtr<T>(ctx);
    auto* out_ptr = GetOutPtr<T>(ctx);
    const auto scalar_operand = GetScalarOperand<T>(ctx);
    const auto n = GetElemCnt(ctx);

    if (out_ptr == in_ptr) {
      RUN_CUDA_KERNEL((ScalarBinaryRightGpuInplace<binary_func, T>), ctx->device_ctx(), n, out_ptr,
                      scalar_operand, n);
    } else {
      RUN_CUDA_KERNEL((ScalarBinaryRightGpu<binary_func, T>), ctx->device_ctx(), n, in_ptr,
                      scalar_operand, out_ptr, n);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(op_name, kernel_type, func_name, kernel_device_type, dtype)         \
  REGISTER_USER_KERNEL(op_name)                                                             \
      .SetCreateFn<                                                                         \
          kernel_type##BinaryKernel<func_name, DeviceType::k##kernel_device_type, dtype>>() \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                          \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);       \
        return ctx.device_type() == DeviceType::k##kernel_device_type                       \
               && y_desc->data_type() == GetDataType<dtype>::value;                         \
      });

#define REGISTER_ADD_KERNEL_WITH_TYPE(type, _) \
  REGISTER_KERNEL("scalar_add", Commutative, BinaryFuncAdd, GPU, type)

OF_PP_FOR_EACH_TUPLE(REGISTER_ADD_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

#undef REGISTER_ADD_KERNEL_WITH_TYPE

#define ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  FLOATING_DATA_TYPE_SEQ

#define REGISTER_MUL_DIV_KERNEL_WITH_TYPE(type, _)                     \
  REGISTER_KERNEL("scalar_mul", Commutative, BinaryFuncMul, GPU, type) \
  REGISTER_KERNEL("left_scalar_div", Left, BinaryFuncDiv, GPU, type)   \
  REGISTER_KERNEL("right_scalar_div", Right, BinaryFuncDiv, GPU, type)

// OF_PP_FOR_EACH_TUPLE(REGISTER_MUL_DIV_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8)

#undef REGISTER_MUL_DIV_KERNEL_WITH_TYPE
#undef ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8

}  // namespace oneflow
