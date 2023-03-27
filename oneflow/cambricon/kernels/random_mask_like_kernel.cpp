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
#include "oneflow/cambricon/cnnl/cnnl_workspace.h"
#include "oneflow/cambricon/ep/mlu_random_generator.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/random_mask_like_kernel.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {
namespace {

class MluRandomMaskLikeKernel final : public user_op::OpKernel {
 public:
  MluRandomMaskLikeKernel() = default;
  ~MluRandomMaskLikeKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kMLU));
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<RandomMaskLikeKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    float rate = ctx->Attr<float>("rate");

    auto* random_mask_like_state = dynamic_cast<RandomMaskLikeKernelState*>(state);
    CHECK_NOTNULL(random_mask_like_state);
    std::shared_ptr<ep::MLUGenerator> generator =
        CHECK_JUST(random_mask_like_state->generator()->Get<ep::MLUGenerator>());
    CHECK_NOTNULL(generator);

    auto cnnl_handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
    // update generator state
    if (generator->need_update_state()) { generator->update_state(cnnl_handle); }

    int64_t count = out->shape_view().elem_cnt();
    CnnlWorkspace workspace(ctx->stream()->As<ep::MluStream>(), count * sizeof(float));

    OF_CNNL_CHECK(cnnlRandGenerateUniform(cnnl_handle, generator->cnnl_rng(), CNNL_DTYPE_FLOAT,
                                          generator->state(), out->shape_view().elem_cnt(), 0.f,
                                          1.f, workspace.dptr()));

    auto primitive = ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
        DeviceType::kMLU, ep::primitive::BinaryOp::kGreaterThan, DataType::kFloat, DataType::kBool,
        1);
    int64_t dims[1] = {count};
    primitive->Launch(ctx->stream(), 1, dims, workspace.dptr(), Scalar(rate), out->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("random_mask_like")
    .SetCreateFn<MluRandomMaskLikeKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kMLU);

}  // namespace
}  // namespace oneflow
