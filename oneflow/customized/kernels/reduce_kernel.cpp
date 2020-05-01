#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace user_op {

template<template<typename> class BinaryFunc, DeviceType device_type, typename T>
class ReduceKernel final : public OpKernel {
 public:
  ReduceKernel() = default;
  ~ReduceKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input_tensor", 0);
    Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("output_tensor", 0);
    Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->GetAttr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        axis.empty() ? Shape::Ones(input_tensor->shape().NumAxes())
                     : CreateReducedShape(input_tensor->shape(), {axis.begin(), axis.end()});
    NdarrayReduce<device_type, T, BinaryFunc>::Reduce(
        ctx->device_ctx(), XpuVarNdarray<T>(reduced_shape, output_tensor->mut_dptr<T>()),
        XpuVarNdarray<const T>(input_tensor->shape(), input_tensor->dptr<T>()),
        XpuVarNdarray<T>(tmp_buffer->shape(), tmp_buffer->mut_dptr<T>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device, typename T>
bool IsMatchedPred(const KernelRegContext& ctx) {
  const TensorDesc* output_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("output_tensor", 0);
  if (ctx.device_type() == device && output_tensor_desc->data_type() == GetDataType<T>::value) {
    return true;
  }
  return false;
}

#define REGISTER_REDUCE_XPU_KERNEL(op_name, binary_func, device, dtype)        \
  REGISTER_USER_KERNEL(op_name)                                                \
      .SetCreateFn<ReduceKernel<binary_func, device, dtype>>()                 \
      .SetIsMatchedPred(IsMatchedPred<device, dtype>)                          \
      .SetInferTmpSizeFn([](InferContext* ctx) {                               \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("input_tensor", 0); \
        return in_shape->elem_cnt() * sizeof(dtype);                           \
      });

#define REGISTER_REDUCE_BY_TYPE(device, dtype)                             \
  REGISTER_REDUCE_XPU_KERNEL("reduce_prod", BinaryFuncProd, device, dtype) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_min", BinaryFuncMin, device, dtype)   \
  REGISTER_REDUCE_XPU_KERNEL("reduce_sum", BinaryFuncSum, device, dtype)

#define REGISTER_REDUCE_KERNEL(device)     \
  REGISTER_REDUCE_BY_TYPE(device, float)   \
  REGISTER_REDUCE_BY_TYPE(device, double)  \
  REGISTER_REDUCE_BY_TYPE(device, int8_t)  \
  REGISTER_REDUCE_BY_TYPE(device, int32_t) \
  REGISTER_REDUCE_BY_TYPE(device, int64_t)

REGISTER_REDUCE_KERNEL(DeviceType::kCPU)
REGISTER_REDUCE_KERNEL(DeviceType::kGPU)

#define REGISTER_REDUCE_LOGICAL_KERNEL(device)                            \
  REGISTER_REDUCE_XPU_KERNEL("reduce_any", BinaryFuncAny, device, int8_t) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_all", BinaryFuncAll, device, int8_t)

REGISTER_REDUCE_LOGICAL_KERNEL(DeviceType::kCPU)
REGISTER_REDUCE_LOGICAL_KERNEL(DeviceType::kGPU)

}  // namespace user_op
}  // namespace oneflow