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

namespace oneflow {

namespace {

template<DeviceType device, typename T>
class BroadcastDivGradKernel final : public user_op::OpKernel {
 public:
  BroadcastDivGradKernel() = default;
  ~BroadcastDivGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* z_tensor = ctx->Tensor4ArgNameAndIndex("z", 0);
    const user_op::Tensor* dz_tensor = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const int64_t num_axes = dz_tensor->shape().NumAxes();
    XpuVarNdarray<const T> dz(dz_tensor->shape(), dz_tensor->dptr<T>(), num_axes);
    XpuVarNdarray<const T> const_tmp(dz.shape(), tmp_buffer->dptr<T>());
    XpuVarNdarray<T> tmp(dz.shape(), tmp_buffer->mut_dptr<T>());

    NdarrayUtil<device, T>::BroadcastDiv(
        ctx->device_ctx(), tmp,
        XpuVarNdarray<const T>(z_tensor->shape(), z_tensor->dptr<T>(), num_axes),
        XpuVarNdarray<const T>(y_tensor->shape(), y_tensor->dptr<T>(), num_axes));
    NdarrayUtil<device, T>::BroadcastMul(ctx->device_ctx(), tmp, dz, const_tmp);
    NdarrayUtil<device, T>::ReduceSum(
        ctx->device_ctx(), XpuVarNdarray<T>(dy_tensor->shape(), dy_tensor->mut_dptr<T>(), num_axes),
        const_tmp, tmp);
    NdarrayUtil<device, T>::InplaceNegative(
        ctx->device_ctx(),
        XpuVarNdarray<T>(dy_tensor->shape(), dy_tensor->mut_dptr<T>(), num_axes));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_BROADCAST_DIV_GRAD_KERNEL(device, dtype_pair)                            \
  REGISTER_USER_KERNEL("broadcast_div_grad")                                              \
      .SetCreateFn<BroadcastDivGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                \
                       & (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(dtype_pair))) \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                        \
        user_op::TensorDesc* z = ctx->TensorDesc4ArgNameAndIndex("z", 0);                 \
        const DataType& data_type = z->data_type();                                       \
        const int64_t elem_cnt = z->shape().elem_cnt();                                   \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type));               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_DIV_GRAD_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_DIV_GRAD_KERNEL, (DeviceType::kGPU),
                                 FLOAT16_DATA_TYPE_SEQ)
#endif

}  // namespace oneflow
