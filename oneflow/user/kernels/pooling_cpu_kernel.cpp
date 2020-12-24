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
#include "oneflow/user/kernels/pooling_kernel_util.h"
#include "oneflow/core/common/eigen_util.h"

namespace oneflow {

namespace user_op{


std::shared_ptr<user_op::OpKernelState> DoCreateKernelState(user_op::KernelInitContext* ctx,
                                                              const int32_t& dim) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::string& padding = ctx->Attr<std::string>("padding");
  const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
  const auto& padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
  const std::vector<int32_t>& dilation = ctx->Attr<std::vector<int32_t>>("dilation");
  const bool return_indices = ctx->Attr<bool>("return_indices");
  const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
  bool is_dynamic = ctx->TensorDesc4ArgNameAndIndex("x", 0)->is_dynamic();
  PoolingParams3D params_3d = PoolingParams3D(
    dim, x_shape, data_format, padding, padding_before, padding_after, 
    kernel_size, stride, dilation, return_indices, ceil_mode
  );
  std::shared_ptr<PoolKernelState> state(new PoolKernelState(params_3d, is_dynamic));
  return std::move(state);
}


template<typename T>
class Maxpool2dCpuKernel final : public user_op::OpKernel {
 public:
  Maxpool2dCpuKernel() = default;
  ~Maxpool2dCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolingCpuKernelUtil<T>::MaxFWCompute(ctx, state);
  };
};

template<typename T>
class Maxpool2dGradCpuKernel final : public user_op::OpKernel {
 public:
  Maxpool2dGradCpuKernel() = default;
  ~Maxpool2dGradCpuKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return DoCreateKernelState(ctx, 2);
  }

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    PoolingCpuKernelUtil<T>::MaxBWCompute(ctx, state);
  };
};




#define REGISTER_POOLING_CPU_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("maxpool_2d")                                                    \
      .SetCreateFn<Maxpool2dCpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                               \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));  \
  REGISTER_USER_KERNEL("maxpool_2d_grad")                                               \
      .SetCreateFn<Maxpool2dGradCpuKernel<dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                               \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); 
  

REGISTER_POOLING_CPU_KERNEL(float)
REGISTER_POOLING_CPU_KERNEL(double)


}  // namespace user_op
}  // namespace oneflow
