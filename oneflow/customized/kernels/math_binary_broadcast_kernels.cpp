#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/customized/kernels/broadcast_grad_util.h"

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
    REGISTER_USER_KERNEL(op_name)                                                \
        .SetCreateFn<MathBinaryBroadcastKernel<binary_func, device, dtype>>()    \
        .SetIsMatchedPred(IsMatchedPred<device, dtype>);

#define REGISTER_BINARYBROADCAST_BY_DEVICETYPE(device, dtype)                              \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_add", BinaryFuncAdd, device, dtype)     \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_sub", BinaryFuncSub, device, dtype)     \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_mul", BinaryFuncMul, device, dtype)     \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_div", BinaryFuncDiv, device, dtype)     \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_minimum", BinaryFuncMin, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_maximum", BinaryFuncMax, device, dtype) \
    REGISTER_BINARYBROADCAST_XPU_KERNEL("broadcast_floor_mod", BinaryFuncFloorMod, device, dtype)

#define REGISTER_BINARYBROADCAST_KERNEL(dtype)                      \
    REGISTER_BINARYBROADCAST_BY_DEVICETYPE(DeviceType::kCPU, dtype) \
    REGISTER_BINARYBROADCAST_BY_DEVICETYPE(DeviceType::kGPU, dtype)

REGISTER_BINARYBROADCAST_KERNEL(float)
REGISTER_BINARYBROADCAST_KERNEL(double)
REGISTER_BINARYBROADCAST_KERNEL(int32_t)
REGISTER_BINARYBROADCAST_KERNEL(int64_t)

template<template<typename, DeviceType> class BinaryGradFunc, DeviceType device_type, typename T>
class MathBinaryBroadcastXGradKernel final : public OpKernel {
    public:
        MathBinaryBroadcastXGradKernel() = default;
        ~MathBinaryBroadcastXGradKernel() = default;

    private:
        void Compute(KernelComputeContext* ctx) const override {
            const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
            const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
            const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
            Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
            Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
            BinaryGradFunc<T, device_type>::XGrad(ctx->device_ctx(), tensor_dz, tensor_dx, tmp_buffer,
                                                  tensor_x, tensor_y);
        }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<template<typename, DeviceType> class BinaryGradFunc, DeviceType device_type, typename T>
class MathBinaryBroadcastYGradKernel final : public OpKernel {
    public:
        MathBinaryBroadcastYGradKernel() = default;
        ~MathBinaryBroadcastYGradKernel() = default;

    private:
        void Compute(KernelComputeContext* ctx) const override {
            const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
            const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
            const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
            Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
            Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
            BinaryGradFunc<T, device_type>::YGrad(ctx->device_ctx(), tensor_dz, tensor_dy, tmp_buffer,
                                                  tensor_x, tensor_y);
        }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BINARYBROADCAST_XGRAD_XPU_KERNEL(op_name, binary_grad_func, device, dtype) \
    REGISTER_USER_KERNEL(op_name)                                                           \
        .SetCreateFn<MathBinaryBroadcastXGradKernel<binary_grad_func, device, dtype>>()     \
        .SetIsMatchedPred(IsMatchedPred<device, dtype>);

#define REGISTER_BINARYBROADCAST_XGRAD_BY_DEVICETYPE(device, dtype)                                    \
    REGISTER_BINARYBROADCAST_XGRAD_XPU_KERNEL("broadcast_add_x_grad", BroadcastAddGrad, device, dtype) \
    REGISTER_BINARYBROADCAST_XGRAD_XPU_KERNEL("broadcast_sub_x_grad", BroadcastSubGrad, device, dtype) \
    REGISTER_BINARYBROADCAST_XGRAD_XPU_KERNEL("broadcast_mul_x_grad", BroadcastMulGrad, device, dtype) \
    REGISTER_BINARYBROADCAST_XGRAD_XPU_KERNEL("broadcast_div_x_grad", BroadcastDivGrad, device, dtype)

#define REGISTER_BINARYBROADCAST_XGRAD_KERNEL(dtype)                      \
    REGISTER_BINARYBROADCAST_XGRAD_BY_DEVICETYPE(DeviceType::kCPU, dtype) \
    REGISTER_BINARYBROADCAST_XGRAD_BY_DEVICETYPE(DeviceType::kGPU, dtype)

REGISTER_BINARYBROADCAST_XGRAD_KERNEL(float)
REGISTER_BINARYBROADCAST_XGRAD_KERNEL(double)
REGISTER_BINARYBROADCAST_XGRAD_KERNEL(int32_t)
REGISTER_BINARYBROADCAST_XGRAD_KERNEL(int64_t)

#define REGISTER_BINARYBROADCAST_YGRAD_XPU_KERNEL(op_name, binary_grad_func, device, dtype) \
    REGISTER_USER_KERNEL(op_name)                                                           \
        .SetCreateFn<MathBinaryBroadcastYGradKernel<binary_grad_func, device, dtype>>()     \
        .SetIsMatchedPred(IsMatchedPred<device, dtype>);

#define REGISTER_BINARYBROADCAST_YGRAD_BY_DEVICETYPE(device, dtype)                                    \
    REGISTER_BINARYBROADCAST_YGRAD_XPU_KERNEL("broadcast_add_y_grad", BroadcastAddGrad, device, dtype) \
    REGISTER_BINARYBROADCAST_YGRAD_XPU_KERNEL("broadcast_sub_y_grad", BroadcastSubGrad, device, dtype) \
    REGISTER_BINARYBROADCAST_YGRAD_XPU_KERNEL("broadcast_mul_y_grad", BroadcastMulGrad, device, dtype) \
    REGISTER_BINARYBROADCAST_YGRAD_XPU_KERNEL("broadcast_div_y_grad", BroadcastDivGrad, device, dtype)

#define REGISTER_BINARYBROADCAST_YGRAD_KERNEL(dtype)                      \
    REGISTER_BINARYBROADCAST_YGRAD_BY_DEVICETYPE(DeviceType::kCPU, dtype) \
    REGISTER_BINARYBROADCAST_YGRAD_BY_DEVICETYPE(DeviceType::kGPU, dtype)

REGISTER_BINARYBROADCAST_YGRAD_KERNEL(float)
REGISTER_BINARYBROADCAST_YGRAD_KERNEL(double)
REGISTER_BINARYBROADCAST_YGRAD_KERNEL(int32_t)
REGISTER_BINARYBROADCAST_YGRAD_KERNEL(int64_t)

}  // namespace user_op
}  // namespace oneflow