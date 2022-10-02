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
#include "oneflow/core/functional/functional.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/distributions/normal_kernel.h"
#include "oneflow/core/ep/include/primitive/softmax.h"
// #include "oneflow/core/ep/include/primitive/one_hot.h"
// #include "oneflow/core/ep/include/primitive/softmax_backward.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Softmax> NewSoftmaxPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::SoftmaxFactory>(ctx->device_type(), data_type);
}

auto SoftmaxPrimitiveExists() {
  return hob::make_custom("SoftmaxPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewSoftmaxPrimitive(&ctx).operator bool();
  });
}

}  //  namespace

template<typename T>
class GumbelSoftmaxKernel final : public user_op::OpKernel {
 public:
  GumbelSoftmaxKernel() = default;
  ~GumbelSoftmaxKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCPU));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    // const auto tau = ctx->Attr<double>("tau");
    // const auto hard = ctx->Attr<bool>("hard");
    CHECK_EQ(in->shape_view().elem_cnt(), out->shape_view().elem_cnt());
    CHECK_EQ(in->data_type(), out->data_type());

    user_op::Tensor* gumbel_noise = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    CHECK_EQ(in->shape_view().elem_cnt(), gumbel_noise->shape_view().elem_cnt());

    // const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    T* gumbel_noise_ptr = gumbel_noise->mut_dptr<T>();

    const ShapeView& in_shape = in->shape_view();
    const int32_t elem_cnt = in_shape.elem_cnt();
    // const int64_t cols = in_shape.At(in_shape.NumAxes() - 1);
    // const int64_t rows = in_shape.Count(0, in_shape.NumAxes() - 1);

    // 1. gumbel_noise = ↓ 
    // 2. tmp = (gumbel_noise + input) / tau √
    // 3. output_soft = softmax(tmp, dim) √
    // 4. (if hard) output = one_hot(output_soft)

    // gumbel_noise
    // 1. generate uniform random TODO(Engine?)
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    auto gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
    std::exponential_distribution<T> dist(1);
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      gumbel_noise_ptr[i] = dist(gen->engine());
    }

    // TODO 2. gumbel_noise = -(exp_dist()).log();
    // FOR_RANGE(int64_t, i, 0, elem_cnt) { gumbel_noise_ptr[i] = -1 * SafeLog(gumbel_noise_ptr[i]); }

    FOR_RANGE(int64_t, i, 0, elem_cnt) { out_ptr[i] = gumbel_noise_ptr[i]; }

    // FOR_RANGE(int64_t, i, 0, elem_cnt) {
    //   out_ptr[i] = (in_ptr[i] + gumbel_noise_ptr[i]) / tau;
    // }

    // std::unique_ptr<ep::primitive::Softmax> primitive = NewSoftmaxPrimitive(ctx);
    // CHECK(primitive);
    // primitive->Launch(ctx->stream(), rows, cols, out_ptr, out->mut_dptr());

    // // TODO: one_hot
    // if (hard) {
      
    // }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GUMBEL_SOFTMAX_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("gumbel_softmax")                              \
      .SetCreateFn<GumbelSoftmaxKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && (SoftmaxPrimitiveExists() == true))         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {             \
        const Shape& in_shape = ctx->InputShape("in", 0);             \
        return in_shape.elem_cnt();                                   \
      });

REGISTER_GUMBEL_SOFTMAX_KERNEL(float)
REGISTER_GUMBEL_SOFTMAX_KERNEL(double)

}  // namespace
