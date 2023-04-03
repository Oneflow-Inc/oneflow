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
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_random_generator.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"

namespace oneflow {

template<typename T>
class MluDropoutKernel final : public user_op::OpKernel {
 public:
  MluDropoutKernel() = default;
  ~MluDropoutKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kMLU));
    return std::make_shared<FusedDropoutKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const float rate = ctx->Attr<float>("rate");

    CnnlTensorDescriptor in_desc(in), out_desc(out);
    CnnlTensorDescriptor mask_desc(mask, CNNL_DTYPE_UINT8);

    auto* dropout_kernel_state = dynamic_cast<FusedDropoutKernelState*>(state);
    CHECK_NOTNULL(dropout_kernel_state);
    std::shared_ptr<ep::MLUGenerator> generator =
        CHECK_JUST(dropout_kernel_state->generator()->Get<ep::MLUGenerator>());

    auto cnnl_handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
    // update generator state
    if (generator->need_update_state()) { generator->update_state(cnnl_handle); }
    OF_CNNL_CHECK(cnnlFusedDropout_v2(cnnl_handle, generator->cnnl_rng(), in_desc.desc(),
                                      in->dptr(), rate, generator->state(), mask_desc.desc(),
                                      mask->mut_dptr(), out_desc.desc(), out->mut_dptr()));

    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      auto bcast_add =
          ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
              ctx->device_type(), ep::primitive::BinaryOp::kAdd, out->data_type(),
              add_to_output->data_type(), out->shape_view().NumAxes());
      CHECK(bcast_add);
      bcast_add->Launch(ctx->stream(), out->shape_view().NumAxes(), out->shape_view().ptr(),
                        out->dptr(), add_to_output->shape_view().NumAxes(),
                        add_to_output->shape_view().ptr(), add_to_output->dptr(), out->mut_dptr());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_DROPOUT_KERNEL(dtype)                                                      \
  REGISTER_USER_KERNEL("dropout")                                                               \
      .SetCreateFn<MluDropoutKernel<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                           \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)         \
                       && (user_op::HobDataType("mask", 0) == GetDataType<bool>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_MLU_DROPOUT_KERNEL(float)

template<typename T>
class MluDropoutGradKernel final : public user_op::OpKernel {
 public:
  MluDropoutGradKernel() = default;
  ~MluDropoutGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale = ctx->Attr<float>("scale");

    CnnlTensorDescriptor dy_desc(dy), dx_desc(dx);
    CnnlTensorDescriptor mask_desc(mask, CNNL_DTYPE_UINT8);

    auto cnnl_handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
    size_t workspace_size = 0;
    OF_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(cnnl_handle, CNNL_MASKED_SCALE, dy_desc.desc(),
                                             mask_desc.desc(), nullptr, dx_desc.desc(),
                                             &workspace_size));
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), workspace_size);
    OF_CNNL_CHECK(cnnlMasked_v4(cnnl_handle, CNNL_MASKED_SCALE, dy_desc.desc(), dy->dptr(),
                                mask_desc.desc(), mask->dptr(), nullptr, nullptr, &scale,
                                workspace.dptr(), workspace_size, dx_desc.desc(), dx->mut_dptr(),
                                nullptr));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_DROPOUT_GRAD_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<MluDropoutGradKernel<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                           \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_MLU_DROPOUT_GRAD_KERNEL(float)

}  // namespace oneflow
