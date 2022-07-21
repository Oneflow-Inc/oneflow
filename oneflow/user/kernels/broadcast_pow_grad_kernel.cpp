/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<DeviceType device, typename T>
class BroadcastPowXGradKernel final : public user_op::OpKernel {
 public:
  BroadcastPowXGradKernel() = default;
  ~BroadcastPowXGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* z_tensor = ctx->Tensor4ArgNameAndIndex("z", 0);
    const user_op::Tensor* dz_tensor = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_axes = dz_tensor->shape_view().NumAxes();
    XpuVarNdarray<const T> dz(dz_tensor->shape_view(), dz_tensor->dptr<T>(), num_axes);
    XpuVarNdarray<const T> y(y_tensor->shape_view(), y_tensor->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device, T>::BroadcastDiv(
        ctx->stream(), tmp,
        XpuVarNdarray<const T>(z_tensor->shape_view(), z_tensor->dptr<T>(), num_axes),
        XpuVarNdarray<const T>(x_tensor->shape_view(), x_tensor->dptr<T>(), num_axes));
    NdarrayUtil<device, T>::BroadcastMul(ctx->stream(), tmp, y, const_tmp);
    NdarrayUtil<device, T>::BroadcastMul(ctx->stream(), tmp, dz, const_tmp);
    NdarrayUtil<device, T>::ReduceSum(
        ctx->stream(),
        XpuVarNdarray<T>(dx_tensor->shape_view(), dx_tensor->mut_dptr<T>(), num_axes), const_tmp,
        tmp);
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device, typename T>
class BroadcastPowYGradKernel final : public user_op::OpKernel {
 public:
  BroadcastPowYGradKernel() = default;
  ~BroadcastPowYGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* z_tensor = ctx->Tensor4ArgNameAndIndex("z", 0);
    const user_op::Tensor* dz_tensor = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const int64_t num_axes = dz_tensor->shape_view().NumAxes();
    const int64_t elem_cnt = z_tensor->shape_view().elem_cnt();
    Memset<device>(ctx->stream(), tmp_buffer->mut_dptr<T>(), 0,
                   GetCudaAlignedSize(elem_cnt * sizeof(T)));
    T* tmp_ptr = tmp_buffer->mut_dptr<T>();
    XpuVarNdarray<const T> z(z_tensor->shape_view(), z_tensor->dptr<T>(), num_axes);
    XpuVarNdarray<const T> dz(dz_tensor->shape_view(), dz_tensor->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());
    XpuVarNdarray<const T> x(x_tensor->shape_view(), x_tensor->dptr<T>(), num_axes);
    XpuVarNdarray<T> dy(dy_tensor->shape_view(), dy_tensor->mut_dptr<T>(), num_axes);
    NdarrayUtil<device, T>::BroadcastAdd(ctx->stream(), tmp, x, const_tmp);
    FOR_RANGE(int64_t, i, 0, elem_cnt) { tmp_ptr[i] = SafeLog(tmp_ptr[i]); }
    NdarrayUtil<device, T>::BroadcastMul(ctx->stream(), tmp, dz, const_tmp);
    NdarrayUtil<device, T>::BroadcastMul(ctx->stream(), tmp, z, const_tmp);
    NdarrayUtil<device, T>::ReduceSum(ctx->stream(), dy, const_tmp, tmp);
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BROADCAST_POW_X_GRAD_KERNEL(device, dtype_pair)                           \
  REGISTER_USER_KERNEL("broadcast_pow_x_grad")                                             \
      .SetCreateFn<BroadcastPowXGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(dtype_pair))) \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                         \
        const user_op::TensorDesc& z = ctx->InputTensorDesc("z", 0);                       \
        const DataType& data_type = z.data_type();                                         \
        const int64_t elem_cnt = z.shape().elem_cnt();                                     \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type));                \
      });

#define REGISTER_BROADCAST_POW_Y_GRAD_KERNEL(device, dtype_pair)                           \
  REGISTER_USER_KERNEL("broadcast_pow_y_grad")                                             \
      .SetCreateFn<BroadcastPowYGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       && (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(dtype_pair))) \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                         \
        const user_op::TensorDesc& z = ctx->InputTensorDesc("z", 0);                       \
        const DataType& data_type = z.data_type();                                         \
        const int64_t elem_cnt = z.shape().elem_cnt();                                     \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type));                \
      });
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_POW_X_GRAD_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_POW_Y_GRAD_KERNEL, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_POW_X_GRAD_KERNEL, (DeviceType::kCUDA),
                                 FLOAT16_DATA_TYPE_SEQ)
#endif

}  // namespace oneflow
