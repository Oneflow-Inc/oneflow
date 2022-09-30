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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ep/npu/npu_stream.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {
template<typename T>
class LayerNormNpuKernel final : public user_op::OpKernel {
 public:
  LayerNormNpuKernel() = default;
  ~LayerNormNpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    float epsilon = ctx->Attr<double>("epsilon");
    const int64_t num_instances = mean->shape().elem_cnt();
    const int64_t norm_size = x->shape().elem_cnt() / num_instances;
    CHECK_EQ(ctx->has_input("gamma", 0), 1); 
    CHECK_EQ(ctx->has_input("beta", 0), 1); 
    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    CHECK_EQ(gamma->shape().elem_cnt(), norm_size);
    user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    int64_t begin_norm_axis = ctx->Attr<int64_t>("begin_norm_axis");
    int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
    NpuCommand npu_command;
    npu_command.OpName("LayerNorm")
            .Input(x)
            .Input(gamma)
            .Input(beta)
            .Output(y)
            .Output(mean)
            .Output(inv_variance)
            .Attr("begin_norm_axis", begin_norm_axis)
            .Attr("begin_params_axis", begin_params_axis)
            .Attr("epsilon", epsilon)
            .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
            .Check();
    npu_command.Run()
            .Realease();
  };
};

#define REGISTER_LAYER_NORM_NPU_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("layer_norm")                                   \
      .SetCreateFn<LayerNormNpuKernel<dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_NPU_KERNEL(float)
REGISTER_LAYER_NORM_NPU_KERNEL(double)
REGISTER_LAYER_NORM_NPU_KERNEL(float16)               

template<typename T>
class LayerNormGradNpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGradNpuKernel() = default;
  ~LayerNormGradNpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    user_op::Tensor* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    const int64_t num_instances = mean->shape().elem_cnt();
    const int64_t norm_size = x->shape().elem_cnt() / num_instances;

    CHECK_EQ(ctx->has_input("gamma", 0), 1); 
    user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);

    NpuCommand npu_command;
    npu_command.OpName("LayerNormGrad")
            .Input(dy)
            .Input(x)
            .Input(inv_variance)
            .Input(mean)
            .Input(gamma)
            .Output(dx)
            .Output(gamma_diff)
            .Output(beta_diff)
            .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
            .Check();
    npu_command.Run()
            .Realease();
  };
};

#define REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("layer_norm_npu_grad")                                              \
      .SetCreateFn<LayerNormGradGpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                     \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))    \
      .SetInplaceProposalFn(                                                               \
          [](const user_op::InferContext& ctx,                                             \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {       \
            if (ctx.has_input("_add_to_output", 0)) {                                      \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true)); \
            }                                                                              \
            return Maybe<void>::Ok();                                                      \
          });


}// namespace oneflow