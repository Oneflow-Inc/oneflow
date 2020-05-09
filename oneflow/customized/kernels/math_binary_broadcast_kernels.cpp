#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace user_op {

template<template<typename> class BinaryFunc, DeviceType device_type, typename T>
class MathBinaryBroadcastKernel final : public OpKernel {
    public:
        MathBinaryBroadcastKernel() = default;
        ~MathBinaryBroadcastKernel() = default;

    private:
        void Compute(KernelComputeContext* ctx) const override {
            const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
            const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
            Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
            // int64_t n = tensor_z->shape().elem_cnt();
            size_t num_axes = tensor_z->shape().NumAxes();
            NdarrayApplyBroadcastBinary<device_type, T, BinaryFunc>::Apply(
                ctx->device_ctx(), XpuVarNdarray<T>(tensor_z->shape(), tensor_z->mut_dptr<T>(), num_axes),
                XpuVarNdarray<const T>(tensor_x->shape(), tensor_x->dptr<T>(), num_axes), 
                XpuVarNdarray<const T>(tensor_y->shape(), tensor_y->dptr<T>(), num_axes));
        }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device, typename T>
bool IsMatchedPred(const KernelRegContext& ctx) {
  const TensorDesc* tensor_z = ctx.TensorDesc4ArgNameAndIndex("z", 0);
  if (ctx.device_type() == device && tensor_z->data_type() == GetDataType<T>::value) {
    return true;
  }
  return false;
}

#define REGISTER_BINARYBROADCAST_XPU_KERNEL(op_name, binary_func, device, dtype) \
    REGISTER_USER_KERNEL(op_name) \
        .SetCreateFn<MathBinaryBroadcastKernel<binary_func, device, dtype>>() \
        .SetIsMatchedPred(IsMatchedPred<device, dtype>);

#define REGISTER_BINARYBROADCAST_BY_DEVICETYPE(device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_add", BinaryFuncAdd, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_sub", BinaryFuncSub, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_mul", BinaryFuncMul, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_div", BinaryFuncDiv, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_minimum", BinaryFuncMin, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_maximum", BinaryFuncMax, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_floor_mod", BinaryFuncFloorMod, device, dtype)

#define REGISTER_BINARYBROADCAST_KERNEL(dtype) \
    REGISTER_BINARYBROADCAST_BY_DEVICETYPE(DeviceType::kCPU, dtype) \
    REGISTER_BINARYBROADCAST_BY_DEVICETYPE(DeviceType::kGPU, dtype)

REGISTER_BINARYBROADCAST_KERNEL(float)
REGISTER_BINARYBROADCAST_KERNEL(double)
REGISTER_BINARYBROADCAST_KERNEL(int32_t)
REGISTER_BINARYBROADCAST_KERNEL(int64_t)

}  // namespace user_op
}  // namespace oneflow