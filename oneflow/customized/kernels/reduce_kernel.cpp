#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/tensor.h"
#include <math.h>
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace user_op {

template<template<typename> class BinaryFunc, typename T>
class ReduceCpuKernel final : public OpKernel {
 public:
  ReduceCpuKernel() = default;
  ~ReduceCpuKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("tensor_in", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("tensor_out", 0);
    Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    auto axis = ctx->GetAttr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        axis.empty() ? Shape::Ones(tensor_in->shape().NumAxes())
                     : CreateReducedShape(tensor_in->shape(), {axis.begin(), axis.end()});
    NdarrayReduce<DeviceType::kCPU, T, BinaryFunc>::Reduce(
        ctx->device_ctx(), XpuVarNdarray<T>(reduced_shape, tensor_out->mut_dptr<T>()),
        XpuVarNdarray<const T>(tensor_in->shape(), tensor_in->dptr<T>()),
        XpuVarNdarray<T>(tmp_buffer->shape(), tmp_buffer->mut_dptr<T>()));
  }
};

template<typename T>
bool IsMatchPred(const KernelRegContext& ctx) {
  const TensorDesc* tensor_out_desc = ctx.TensorDesc4ArgNameAndIndex("tensor_out", 0);
  if (ctx.device_type() == DeviceType::kCPU
      && tensor_out_desc->data_type() == GetDataType<T>::value) {
    return true;
  }
  return false;
}

#define REGISTER_REDUCE_CPU_KERNEL(op_name, binary_func, dtype)             \
  REGISTER_USER_KERNEL(op_name)                                             \
      .SetCreateFn<ReduceCpuKernel<binary_func, dtype>>()                   \
      .SetIsMatchedPred(IsMatchPred<dtype>)                                 \
      .SetInferTmpSizeFn([](InferContext* ctx) {                            \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("tensor_in", 0); \
        return in_shape->elem_cnt() * sizeof(dtype);                        \
      });

REGISTER_REDUCE_CPU_KERNEL("reduce_prod", BinaryFuncProd, float)
REGISTER_REDUCE_CPU_KERNEL("reduce_prod", BinaryFuncProd, double)
REGISTER_REDUCE_CPU_KERNEL("reduce_prod", BinaryFuncProd, int32_t)
REGISTER_REDUCE_CPU_KERNEL("reduce_prod", BinaryFuncProd, int64_t)

REGISTER_REDUCE_CPU_KERNEL("reduce_min", BinaryFuncMin, float)
REGISTER_REDUCE_CPU_KERNEL("reduce_min", BinaryFuncMin, double)
REGISTER_REDUCE_CPU_KERNEL("reduce_min", BinaryFuncMin, int32_t)
REGISTER_REDUCE_CPU_KERNEL("reduce_min", BinaryFuncMin, int64_t)

REGISTER_REDUCE_CPU_KERNEL("reduce_any", BinaryFuncAny, float)
REGISTER_REDUCE_CPU_KERNEL("reduce_any", BinaryFuncAny, double)
REGISTER_REDUCE_CPU_KERNEL("reduce_any", BinaryFuncAny, int32_t)
REGISTER_REDUCE_CPU_KERNEL("reduce_any", BinaryFuncAny, int64_t)

}  // namespace user_op
}  // namespace oneflow