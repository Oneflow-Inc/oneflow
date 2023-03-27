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
#include "oneflow/cambricon/cnnl/cnnl_types.h"
#include "oneflow/cambricon/ep/mlu_random_generator.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

template<typename T>
class MluNormalKernel final : public user_op::OpKernel {
 public:
  MluNormalKernel() = default;
  ~MluNormalKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kMLU));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const double mean = ctx->Attr<double>("mean");
    const double std = ctx->Attr<double>("std");

    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    std::shared_ptr<ep::MLUGenerator> generator =
        CHECK_JUST(distribution_state->generator()->Get<ep::MLUGenerator>());
    CHECK_NOTNULL(generator);

    auto cnnl_handle = ctx->stream()->As<ep::MluStream>()->cnnl_handle();
    // update generator state
    if (generator->need_update_state()) { generator->update_state(cnnl_handle); }

    cnnlDataType_t cnnl_dtype = ConvertToCnnlDataType(GetDataType<T>::value);
    OF_CNNL_CHECK(cnnlRandGenerateNormal(cnnl_handle, generator->cnnl_rng(), cnnl_dtype,
                                         generator->state(), out->shape_view().elem_cnt(), mean,
                                         std, out->mut_dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_NORMAL_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("normal").SetCreateFn<MluNormalKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kMLU)                                    \
      && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

REGISTER_MLU_NORMAL_KERNEL(float16)
REGISTER_MLU_NORMAL_KERNEL(float)

}  // namespace
}  // namespace oneflow
