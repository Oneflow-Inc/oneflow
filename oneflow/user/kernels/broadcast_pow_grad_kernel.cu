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
#include "oneflow/core/kernel/kernel_util.cuh"
namespace oneflow {

namespace {
template<DeviceType device>
class BroadcastPowYGradKernel<device, float16> final : public user_op::OpKernel {
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

    const int64_t num_axes = dz_tensor->shape().NumAxes();
    const int64_t elem_cnt = z_tensor->shape().elem_cnt();
    Memset<device>(ctx->device_ctx(), tmp_buffer->mut_dptr<float16>(), 0,
                   GetCudaAlignedSize(elem_cnt * sizeof(T)));
    half* tmp_ptr = reinterpret_cast<half*>(tmp_buffer->mut_dptr<float16>());
    XpuVarNdarray<const float16> z(z_tensor->shape(), z_tensor->dptr<float16>(), num_axes);
    XpuVarNdarray<const float16> dz(dz_tensor->shape(), dz_tensor->dptr<float16>(), num_axes);
    XpuVarNdarray<const float16> const_tmp(dz.shape(), tmp_buffer->dptr<float16>());
    XpuVarNdarray<float16> tmp(dz.shape(), tmp_buffer->mut_dptr<float16>());
    XpuVarNdarray<const float16> x(x_tensor->shape(), x_tensor->dptr<float16>(), num_axes);
    XpuVarNdarray<float16> dy(dy_tensor->shape(), dy_tensor->mut_dptr<float16>(), num_axes);
    NdarrayUtil<device, float16>::BroadcastAdd(ctx->device_ctx(), tmp, x, const_tmp);
    FOR_RANGE(int64_t, i, 0, elem_cnt) { tmp_ptr[i] = SafeLog(tmp_ptr[i]); }
    NdarrayUtil<device, float16>::BroadcastMul(ctx->device_ctx(), tmp, dz, const_tmp);
    NdarrayUtil<device, float16>::BroadcastMul(ctx->device_ctx(), tmp, z, const_tmp);
    NdarrayUtil<device, float16>::ReduceSum(ctx->device_ctx(), dy, const_tmp, tmp);
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
}  // namespace
#define REGISTER_BROADCAST_POW_Y_GRAD_KERNEL(device, dtype_pair)                          \
  REGISTER_USER_KERNEL("broadcast_pow_y_grad")                                            \
      .SetCreateFn<BroadcastPowYGradKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                \
                       & (user_op::HobDataType("x", 0) == OF_PP_PAIR_SECOND(dtype_pair))) \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                        \
        const user_op::TensorDesc& z = ctx->InputTensorDesc("z", 0);                      \
        const DataType& data_type = z.data_type();                                        \
        const int64_t elem_cnt = z.shape().elem_cnt();                                    \
        return GetCudaAlignedSize(elem_cnt * GetSizeOfDataType(data_type));               \
      });
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BROADCAST_POW_Y_GRAD_KERNEL, (DeviceType::kGPU),
                                 FLOAT16_DATA_TYPE_SEQ)
}  // namespace oneflow
