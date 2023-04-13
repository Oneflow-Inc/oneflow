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
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/core/ep/include/primitive/add.h"

namespace oneflow {

namespace {

template<typename T>
void MaskAndScale(ep::Stream* stream, const int64_t n, float scale, const T* x, const bool* mask,
                  T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
}

template<typename T>
void FusedDropoutKernel(ep::Stream* stream, const int64_t elem_cnt,
                        const std::shared_ptr<one::CPUGeneratorImpl>& cpu_gen, const float rate,
                        float scale, const T* x, bool* mask, T* y) {
  /*
  `uniform_real_distribution` interval is [a, b).
  And `curand_uniform4` interval is (0, 1.0], so we use > in CUDA and use >= in CPU.
  */
  std::uniform_real_distribution<float> random_distribution(GetZeroVal<float>(),
                                                            GetOneVal<float>());
  for (int64_t i = 0; i < elem_cnt; ++i) {
    mask[i] = random_distribution(cpu_gen->engine()) >= rate;
    y[i] = x[i] * static_cast<T>(mask[i]) * scale;
  }
}

template<typename T>
class DropoutKernelCPU final : public user_op::OpKernel {
 public:
  DropoutKernelCPU() = default;
  ~DropoutKernelCPU() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(kCPU));
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<FusedDropoutKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const float rate = ctx->Attr<float>("rate");
    float scale = 0.0f;
    if (rate < 1.0f) { scale = 1.0f / (1.0f - rate); }

    auto* fused_dropout_kernel_state = dynamic_cast<FusedDropoutKernelState*>(state);
    CHECK_NOTNULL(fused_dropout_kernel_state);
    const auto& generator = fused_dropout_kernel_state->generator();
    CHECK_NOTNULL(generator);
    std::shared_ptr<one::CPUGeneratorImpl> cpu_generator =
        CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());

    FusedDropoutKernel<T>(ctx->stream(), in->shape_view().elem_cnt(), cpu_generator, rate, scale,
                          in->dptr<T>(), mask->mut_dptr<bool>(), out->mut_dptr<T>());

    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), out->data_type());
      CHECK_EQ(add_to_output->shape_view(), out->shape_view());
      std::unique_ptr<ep::primitive::Add> primitive =
          ep::primitive::NewPrimitive<ep::primitive::AddFactory>(DeviceType::kCPU,
                                                                 add_to_output->data_type());
      CHECK(primitive);
      primitive->Launch(ctx->stream(), out->dptr<T>(), add_to_output->dptr<T>(), out->mut_dptr<T>(),
                        add_to_output->shape_view().elem_cnt());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_CPU(dtype)                                                      \
  REGISTER_USER_KERNEL("dropout")                                                               \
      .SetCreateFn<DropoutKernelCPU<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)         \
                       && (user_op::HobDataType("mask", 0) == GetDataType<bool>::value))        \
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
    MaskAndScale<T>(ctx->stream(), dy->shape_view().elem_cnt(), scale, dy->dptr<T>(),
                    mask->dptr<bool>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_CPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelCPU<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_CPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_CPU(double)

}  // namespace
}  // namespace oneflow
