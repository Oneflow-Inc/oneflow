#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct AddByScalar final {
  using func = void (*)(DeviceCtx*, const int64_t, const T*, const T, T*);
  constexpr static func Invoke = NewKernelUtil<device_type>::AddByScalar;
};

template<DeviceType device_type, typename T>
struct MulByScalar final {
  using func = void (*)(DeviceCtx*, const int64_t, const T*, const T, T*);
  constexpr static func Invoke = NewKernelUtil<device_type>::MulByScalar;
};

template<DeviceType device_type, typename T>
struct TensorDivByScalar final {
  using func = void (*)(DeviceCtx*, const int64_t, const T*, const T, T*);
  constexpr static func Invoke = NewKernelUtil<device_type>::TensorDivByScalar;
};

template<DeviceType device_type, typename T>
struct ScalarDivByTensor final {
  using func = void (*)(DeviceCtx*, const int64_t, const T*, const T, T*);
  constexpr static func Invoke = NewKernelUtil<device_type>::ScalarDivByTensor;
};

// TODO: question: 交换律在哪里体现
template<template<DeviceType, typename T> class func, DeviceType device_type, typename T>
class ComutativeBinaryKernel final : public user_op::OpKernel {
 public:
  ComutativeBinaryKernel() = default;
  ~ComutativeBinaryKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->GetAttr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->GetAttr<int64_t>("int_operand"));
    } else if (ctx->GetAttr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->GetAttr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    func<device_type, T>::Invoke(ctx->device_ctx(), out->shape().elem_cnt(), in_ptr, scalar_operand,
                                 out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(op_name, func_name, kernel_device_type, dtype)                            \
  REGISTER_USER_KERNEL(OF_PP_STRINGIZE(op_name))                                                  \
      .SetCreateFn<ComutativeBinaryKernel<func_name, DeviceType::k##kernel_device_type, dtype>>() \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);             \
        return ctx.device_type() == DeviceType::k##kernel_device_type                             \
               && y_desc->data_type() == GetDataType<dtype>::value;                               \
      });

#define REGISTER_ADD_KERNEL_WITH_TYPE(type, _)        \
  REGISTER_KERNEL(scalar_add, AddByScalar, CPU, type) \
  REGISTER_KERNEL(scalar_add, AddByScalar, GPU, type)

OF_PP_FOR_EACH_TUPLE(REGISTER_ADD_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

#undef REGISTER_ADD_KERNEL_WITH_TYPE

#define ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  FLOATING_DATA_TYPE_SEQ

#define REGISTER_MUL_DIV_KERNEL_WITH_TYPE(type, _)                    \
  REGISTER_KERNEL(scalar_mul, MulByScalar, CPU, type)                 \
  REGISTER_KERNEL(scalar_mul, MulByScalar, GPU, type)                 \
  REGISTER_KERNEL(scalar_div_left_scalar, ScalarDivByTensor, CPU, type)   \
  REGISTER_KERNEL(scalar_div_left_scalar, ScalarDivByTensor, GPU, type)   \
  REGISTER_KERNEL(scalar_div_right_scalar, TensorDivByScalar, CPU, type) \
  REGISTER_KERNEL(scalar_div_right_scalar, TensorDivByScalar, GPU, type)

OF_PP_FOR_EACH_TUPLE(REGISTER_MUL_DIV_KERNEL_WITH_TYPE, ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8)

#undef REGISTER_MUL_DIV_KERNEL_WITH_TYPE
#undef ARITHMETIC_DATA_TYPE_SEQ_WITHOUT_INT8

}  // namespace oneflow
