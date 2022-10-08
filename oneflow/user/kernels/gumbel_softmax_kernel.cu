
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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/uniform_distribution.h"
#include "oneflow/core/ep/include/primitive/softmax.h"
// #include "oneflow/core/ep/include/primitive/softmax_backward.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GumbelSoftmaxAddNoiseForwardGpu(const int n, const float tau, const T* in,
                                                const T* gumbel_noise, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = (in[i] + gumbel_noise[i]) / tau; }
}

template<typename T>
__global__ void GumbelSoftmaxNoiseFromUniformGpu(const int n, const T* gumbel_noise, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = static_cast<T>(-1.0) * SafeLog(static_cast<T>(-1.0) * SafeLog(
      static_cast<T>(1.0) - gumbel_noise[i]
    ));
  }
}

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

template<DeviceType device_type, typename T>
class GpuGumbelSoftmaxKernel final : public user_op::OpKernel {
 public:
  GpuGumbelSoftmaxKernel() = default;
  ~GpuGumbelSoftmaxKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
    const auto seed =
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed")));
    generator->set_current_seed(seed);
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
    CHECK_EQ(in->shape_view().elem_cnt(), gumbel_noise->shape_view().elem_cnt());

    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    T* gumbel_noise_ptr = gumbel_noise->mut_dptr<T>();

    const ShapeView& in_shape = in->shape_view();
    const int32_t elem_cnt = in_shape.elem_cnt();
    const int64_t cols = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t rows = in_shape.Count(0, in_shape.NumAxes() - 1);

    // gumbel_noise
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    UniformDistribution<device_type, T> distribution(static_cast<T>(0.0), static_cast<T>(1.0));
    distribution(ctx->stream(), elem_cnt, gumbel_noise_ptr, generator);
    RUN_CUDA_KERNEL((GumbelSoftmaxNoiseFromUniformGpu<T>), ctx->stream(), elem_cnt, elem_cnt,
                    gumbel_noise_ptr, gumbel_noise->mut_dptr<T>());

    RUN_CUDA_KERNEL((GumbelSoftmaxAddNoiseForwardGpu<T>), ctx->stream(), elem_cnt, elem_cnt, tau,
                    in_ptr, gumbel_noise->mut_dptr<T>(), out_ptr);

    std::unique_ptr<ep::primitive::Softmax> primitive = NewSoftmaxPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), rows, cols, out_ptr, out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_GUMBEL_SOFTMAX_KERNEL(device, dtype)                                \
  REGISTER_USER_KERNEL("gumbel_softmax")                                             \
      .SetCreateFn<GpuGumbelSoftmaxKernel<device, dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDeviceType() == device)                       \
                       && (SoftmaxPrimitiveExists() == true))                        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                            \
        const Shape& in_shape = ctx->InputShape("in", 0);                            \
        return in_shape.elem_cnt();                                                  \
      });

REGISTER_GPU_GUMBEL_SOFTMAX_KERNEL(DeviceType::kCUDA, float)
REGISTER_GPU_GUMBEL_SOFTMAX_KERNEL(DeviceType::kCUDA, double)

}  //  oneflow