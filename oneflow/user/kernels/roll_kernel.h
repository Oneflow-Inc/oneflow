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
#ifndef ONEFLOW_USER_KERNELS_ROLL_KERNEL_H_
#define ONEFLOW_USER_KERNELS_ROLL_KERNEL_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct RollChange {
  static void Invoke(DeviceCtx *ctx, std::vector<int32_t> move, oneflow::fixed_vector<long int, 20> dim, 
                   const T *x, T *y);
};

template<DeviceType device_type, typename T>
class RollKernel final : public user_op::OpKernel {
public: 
    RollKernel() = default;
    ~RollKernel() override = default;

private:
    void Compute(user_op::KernelComputeContext* ctx) const override {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
        const user_op::TensorDesc* in_shape = ctx->TensorDesc4ArgNameAndIndex("in", 0);
        const oneflow::fixed_vector<long int, 20> in_dim_vec = in_shape->shape().dim_vec();
        const std::vector<int32_t> move = ctx->Attr<std::vector<int32_t>>("shifts");
        RollChange<device_type, T>::Invoke(ctx->device_ctx(),
           move,
           in_dim_vec,
           in->dptr<T>(),
           out->mut_dptr<T>());
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ROLL_USER_KERNEL(device, dtype)                                                             \
  REGISTER_USER_KERNEL("roll")                                                                                     \
      .SetCreateFn<RollKernel<device, dtype>>()                                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) & \
                        user_op::HobDataType("in", 0) == GetDataType<dtype>::value);

} // namespace oneflow
#endif // ONEFLOW_USER_KERNELS_ROLL_KERNEL_H_
