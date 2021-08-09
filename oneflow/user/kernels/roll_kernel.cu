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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

template <typename T>
void Roll(DeviceCtx *ctx, std::vector<int32_t> move, oneflow::fixed_vector<long int, 20> dim, const T *x, T *y) {

    int32_t len = dim[0];
    int32_t width = dim[1];
    int32_t shift = move[0] % len;
    int32_t cpyFirst = width*(len-shift);
    cudaMemcpy(y, x + width*sizeof(T), cpyFirst*sizeof(T), cudaMemcpyDefault);
    int32_t cpySec = width*shift;
    cudaMemcpy(y+cpyFirst, x, cpySec*sizeof(T), cudaMemcpyDefault);

}

template <typename T>
void RollGrad(DeviceCtx *ctx, std::vector<int32_t> move, oneflow::fixed_vector<long int, 20> dim, const T *x, T *y) {

    int32_t len = dim[0];
    int32_t width = dim[1];
    int32_t shift = move[0] % len;
    int32_t cpyFirst = width*shift;
    int32_t cpySec = width*(len-shift);
    cudaMemcpy(y, x + width*sizeof(T), cpyFirst*sizeof(T), cudaMemcpyDefault);    
    cudaMemcpy(y+cpyFirst, x, cpySec*sizeof(T), cudaMemcpyDefault);

}

template<typename T>
class GpuRollKernel final : public user_op::OpKernel {
public: 
    GpuRollKernel() = default;
    ~GpuRollKernel() override = default;

private:
    void Compute(user_op::KernelComputeContext* ctx) const override {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
        const user_op::TensorDesc* in_shape = ctx->TensorDesc4ArgNameAndIndex("in", 0);
        const oneflow::fixed_vector<long int, 20> in_dim_vec = in_shape->shape().dim_vec();
        const std::vector<int32_t> move = ctx->Attr<std::vector<int32_t>>("shifts");
        Roll<T>(ctx->device_ctx(),
           move,
           in_dim_vec,
           in->dptr<T>(),
           out->mut_dptr<T>());
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_GPU_ROLL_KERNEL(dtype)                                                             \
  REGISTER_USER_KERNEL("roll")                                                                                     \
      .SetCreateFn<GpuRollKernel<dtype>>()                                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU))
  
REGISTER_GPU_ROLL_KERNEL(float);
REGISTER_GPU_ROLL_KERNEL(double);
#undef REGISTER_GPU_ROLL_KERNEL

template<typename T>
class GpuRollGradKernel final : public user_op::OpKernel {
public: 
    GpuRollGradKernel() = default;
    ~GpuRollGradKernel() override = default;

private:
    void Compute(user_op::KernelComputeContext* ctx) const override {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
        const user_op::TensorDesc* in_shape = ctx->TensorDesc4ArgNameAndIndex("in", 0);
        const oneflow::fixed_vector<long int, 20> in_dim_vec = in_shape->shape().dim_vec();
        const std::vector<int32_t> move = ctx->Attr<std::vector<int32_t>>("shifts");
        RollGrad<T>(ctx->device_ctx(),
           move,
           in_dim_vec,
           in->dptr<T>(),
           out->mut_dptr<T>());
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ROLL_GRAD_KERNEL(dtype)                                                             \
  REGISTER_USER_KERNEL("roll_grad")                                                                                     \
      .SetCreateFn<GpuRollGradKernel<dtype>>()                                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU))
  
REGISTER_GPU_ROLL_GRAD_KERNEL(float);
REGISTER_GPU_ROLL_GRAD_KERNEL(double);
#undef REGISTER_GPU_ROLL_GRAD_KERNEL

}
}
