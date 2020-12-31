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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/broadcast_maximum_kernel_util.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
class BroadcastMaximumBackward final : public user_op::OpKernel {
 public:
  BroadcastMaximumBackward() = default;
  ~BroadcastMaximumBackward() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
    user_op::Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);

    const T* dptr_dz = tensor_dz->dptr<T>();
    const T* dptr_x = tensor_x->dptr<T>();
    const T* dptr_y = tensor_y->dptr<T>();

    T* dptr_dx = tensor_dx->mut_dptr<T>();
    T* dptr_dy = tensor_dy->mut_dptr<T>();

    MaximumBackwardFunctor<device_type, T>()(ctx->device_ctx(), tensor_dz->shape().elem_cnt(),
                                             dptr_dz, dptr_x, dptr_y, dptr_dx, dptr_dy);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BW_MAXIMUM_KERNEL(device, dtype)             \
  REGISTER_USER_KERNEL("broadcast_maximum_backward")          \
      .SetCreateFn<BroadcastMaximumBackward<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)    \
                       & (user_op::HobDataType("dz", 0) == GetDataType<dtype>::value));

REGISTER_BW_MAXIMUM_KERNEL(DeviceType::kCPU, float);
REGISTER_BW_MAXIMUM_KERNEL(DeviceType::kCPU, double);
REGISTER_BW_MAXIMUM_KERNEL(DeviceType::kGPU, float);
REGISTER_BW_MAXIMUM_KERNEL(DeviceType::kGPU, double);

REGISTER_USER_OP_GRAD("broadcast_maximum")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      const auto& grad_op_func = [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("broadcast_maximum_backward")
            .InputBind("dz", ctx->FwOp().output_grad("z", 0))
            .InputBind("x", ctx->FwOp().input("x", 0))
            .InputBind("y", ctx->FwOp().input("y", 0))
            .Output("dx")
            .Output("dy")
            .Build();
      };
      ctx->DefineOp(grad_op_name, grad_op_func);
      const auto& dx_get_func = [&ctx, &grad_op_name]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("dx", 0);
      };
      const auto& dy_get_func = [&ctx, &grad_op_name]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("dy", 0);
      };
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), dx_get_func);
      ctx->FwOp().InputGradBind(user_op::OpArg("y", 0), dy_get_func);

    });
}  // namespace user_op
}  // namespace oneflow
