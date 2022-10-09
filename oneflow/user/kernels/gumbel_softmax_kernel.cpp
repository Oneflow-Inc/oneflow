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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/uniform_distribution.h"
#include "oneflow/user/kernels/gumbel_softmax_kernel_util.h"

namespace oneflow {

namespace {

auto SoftmaxPrimitiveExists() {
  return hob::make_custom("SoftmaxPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewSoftmaxPrimitive(&ctx).operator bool();
  });
}

auto SoftmaxBackwardPrimitiveExists() {
  return hob::make_custom("SoftmaxBackwardPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewSoftmaxBackwardPrimitive(&ctx).operator bool();
                          });
}

}  //  namespace

template<DeviceType device_type, typename T>
class GumbelSoftmaxKernel final : public user_op::OpKernel {
 public:
  GumbelSoftmaxKernel() = default;
  ~GumbelSoftmaxKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
    // const auto seed =
        // CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed")));
    // generator->set_current_seed(seed);
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto tau = ctx->Attr<double>("tau");
    CHECK_EQ(in->shape_view().elem_cnt(), out->shape_view().elem_cnt());
    CHECK_EQ(in->data_type(), out->data_type());
 
    user_op::Tensor* gumbel_noise = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    T* gumbel_noise_ptr = gumbel_noise->mut_dptr<T>();

    const ShapeView& in_shape = in->shape_view();
    const int64_t elem_cnt = in_shape.elem_cnt();
    const int64_t cols = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t rows = in_shape.Count(0, in_shape.NumAxes() - 1);

    // gumbel_noise
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    UniformDistribution<device_type, T> distribution(static_cast<T>(0.0), static_cast<T>(1.0));
    distribution(ctx->stream(), elem_cnt, gumbel_noise_ptr, generator);

    GumbelSoftmaxAddNoiseImpl<device_type, T>::Forward(ctx->stream(), tau, elem_cnt, in_ptr,
                                                       gumbel_noise_ptr, out_ptr);

    std::unique_ptr<ep::primitive::Softmax> primitive = NewSoftmaxPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), rows, cols, out_ptr, out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GUMBEL_SOFTMAX_KERNEL(device, dtype)                 \
  REGISTER_USER_KERNEL("gumbel_softmax")                              \
      .SetCreateFn<GumbelSoftmaxKernel<device, dtype>>()              \
      .SetIsMatchedHob((user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDeviceType() == device)        \
                       && (SoftmaxPrimitiveExists() == true))         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {             \
        const Shape& in_shape = ctx->InputShape("in", 0);             \
        return in_shape.elem_cnt() * sizeof(dtype);                   \
      });

REGISTER_GUMBEL_SOFTMAX_KERNEL(DeviceType::kCPU, float)
REGISTER_GUMBEL_SOFTMAX_KERNEL(DeviceType::kCPU, double)
REGISTER_GUMBEL_SOFTMAX_KERNEL(DeviceType::kCUDA, float)
REGISTER_GUMBEL_SOFTMAX_KERNEL(DeviceType::kCUDA, double)

class GumbelSoftmaxGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GumbelSoftmaxGradKernel() = default;
  ~GumbelSoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t num_classes = y->shape_view().At(y->shape_view().NumAxes() - 1);
    const int64_t num_instances = y->shape_view().elem_cnt() / num_classes;

    std::unique_ptr<ep::primitive::SoftmaxBackward> primitive = NewSoftmaxBackwardPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), num_instances, num_classes, y->dptr(), dy->dptr(),
                      dx->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("gumbel_softmax_grad")
    .SetCreateFn<GumbelSoftmaxGradKernel>()
    .SetIsMatchedHob(SoftmaxBackwardPrimitiveExists() == true);

}  // oneflow
