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
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/primitive/include/add.h"

namespace oneflow {

namespace {

template<typename T>
void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x, const int8_t* mask,
                  T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
}

template<typename T>
class DropoutKernelCPU final : public user_op::OpKernel {
 public:
  DropoutKernelCPU() = default;
  ~DropoutKernelCPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const float scale = ctx->Attr<float>("scale");
    MaskAndScale<T>(ctx->device_ctx(), in->shape().elem_cnt(), scale, in->dptr<T>(),
                    mask->dptr<int8_t>(), out->mut_dptr<T>());
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), out->data_type());
      CHECK_EQ(add_to_output->shape(), out->shape());
      std::unique_ptr<primitive::Add> primitive = primitive::NewPrimitive<primitive::AddFactory>(
          DeviceType::kCPU, add_to_output->data_type());
      CHECK(primitive);
      primitive->Launch(ctx->stream_ctx(), add_to_output->dptr<T>(), out->dptr<T>(),
                        out->mut_dptr<T>(), add_to_output->shape().elem_cnt());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_CPU(dtype)                                                      \
  REGISTER_USER_KERNEL("dropout")                                                               \
      .SetCreateFn<DropoutKernelCPU<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                       \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_KERNEL_CPU(float)
REGISTER_DROPOUT_KERNEL_CPU(double)

template<typename T>
class DropoutGradKernelCPU final : public user_op::OpKernel {
 public:
  DropoutGradKernelCPU() = default;
  ~DropoutGradKernelCPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale = ctx->Attr<float>("scale");
    MaskAndScale<T>(ctx->device_ctx(), dy->shape().elem_cnt(), scale, dy->dptr<T>(),
                    mask->dptr<int8_t>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_CPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelCPU<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                       \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_CPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_CPU(double)

}  // namespace
}  // namespace oneflow
