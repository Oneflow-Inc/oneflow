#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/tensor.h"
#include <math.h>
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
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("tensor_in", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("tensor_out", 0);
    Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto& axis = ctx->GetAttr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        axis.empty() ? Shape::Ones(tensor_in->shape().NumAxes())
                     : CreateReducedShape(tensor_in->shape(), {axis.begin(), axis.end()});
    NdarrayReduce<device_type, T, BinaryFunc>::Reduce(
        ctx->device_ctx(), XpuVarNdarray<T>(reduced_shape, tensor_out->mut_dptr<T>()),
        XpuVarNdarray<const T>(tensor_in->shape(), tensor_in->dptr<T>()),
        XpuVarNdarray<T>(tmp_buffer->shape(), tmp_buffer->mut_dptr<T>()));
  }
};

template<DeviceType device, typename T>
bool IsMatchedPred(const KernelRegContext& ctx) {
  const TensorDesc* tensor_out_desc = ctx.TensorDesc4ArgNameAndIndex("tensor_out", 0);
  if (ctx.device_type() == device && tensor_out_desc->data_type() == GetDataType<T>::value) {
    return true;
  }
  return false;
}

#define REGISTER_REDUCE_XPU_KERNEL(op_name, binary_func, device, dtype)     \
  REGISTER_USER_KERNEL(op_name)                                             \
      .SetCreateFn<ReduceKernel<binary_func, device, dtype>>()              \
      .SetIsMatchedPred(IsMatchedPred<device, dtype>)                       \
      .SetInferTmpSizeFn([](InferContext* ctx) {                            \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("tensor_in", 0); \
        return in_shape->elem_cnt() * sizeof(dtype);                        \
      });

#define REGISTER_REDUCE_BY_DEVICETYPE(device, dtype)                       \
  REGISTER_REDUCE_XPU_KERNEL("reduce_prod", BinaryFuncProd, device, dtype) \
  REGISTER_REDUCE_XPU_KERNEL("reduce_min", BinaryFuncMin, device, dtype)   \
  REGISTER_REDUCE_XPU_KERNEL("reduce_any", BinaryFuncAny, device, dtype)

#define REGISTER_REDUCE_KERNEL(dtype)                    \
  REGISTER_REDUCE_BY_DEVICETYPE(DeviceType::kCPU, dtype) \
  REGISTER_REDUCE_BY_DEVICETYPE(DeviceType::kGPU, dtype)

REGISTER_REDUCE_KERNEL(float)
REGISTER_REDUCE_KERNEL(double)
REGISTER_REDUCE_KERNEL(int32_t)
REGISTER_REDUCE_KERNEL(int64_t)

}  // namespace user_op
}  // namespace oneflow