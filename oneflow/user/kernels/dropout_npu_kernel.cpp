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
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/core/ep/include/primitive/add.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

namespace {


template<typename T>
class DropoutKernelNPU final : public user_op::OpKernel {
 public:
  DropoutKernelNPU() = default;
  ~DropoutKernelNPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    float rate_fp32 = 1.0f - ctx->Attr<float>("rate");
    T rate(rate_fp32);
    DimVector shape_dim_v;
    out->shape_view().ToDimVector(&shape_dim_v);
    std::vector<int> shape_vector;
    for(auto sh:shape_dim_v)
    {
        shape_vector.push_back(sh);
    }
    std::vector<int64_t> shape_desc = {static_cast<int>(shape_vector.size())};
    HostTensorWrapper shape_wrap(ACL_INT32, ACL_FORMAT_ND, shape_desc.size(), shape_desc.data(),
                            shape_vector.size()*sizeof(int), shape_vector.data());
    
    // use 0 and nullptr for scalar
    HostTensorWrapper rate_wrap(DataTypeTraits<T>::type, ACL_FORMAT_ND, 0, nullptr,
                            sizeof(T), &rate, ACL_MEMTYPE_HOST);    
    NpuCommand npu_command;
    npu_command.OpName("DropOutGenMask")
               .Input(shape_wrap)
               .Input(rate_wrap)
               .Attr("seed",(int64_t)0)
               .Attr("seed2",(int64_t)0)
               .Output(mask)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();

    NpuCommand do_command;
    do_command.OpName("DropOutDoMask")
               .Input(in)
               .Input(mask)
               .Input(rate_wrap)
               .Output(out)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    do_command.Run()
               .Realease();

    if (ctx->has_input("_add_to_output", 0)) {
        UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_NPU(dtype)                                                      \
  REGISTER_USER_KERNEL("dropout_npu")                                                               \
      .SetCreateFn<DropoutKernelNPU<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                           \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_KERNEL_NPU(float16)
REGISTER_DROPOUT_KERNEL_NPU(float)

template<typename T>
class DropoutGradKernelNPU final : public user_op::OpKernel {
 public:
  DropoutGradKernelNPU() = default;
  ~DropoutGradKernelNPU() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    float scale = ctx->Attr<float>("scale");
    float rate_fp32 = 1.0f/scale;
    T rate(rate_fp32);
    HostTensorWrapper rate_wrap(DataTypeTraits<T>::type, ACL_FORMAT_ND, 0, nullptr,
                            sizeof(T), &rate);    

    NpuCommand npu_command;
    npu_command.OpName("DropOutDoMask")
               .Input(dy)
               .Input(mask)
               .Input(rate_wrap)
               .Output(dx)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_NPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelNPU<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                           \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_NPU(float16)
REGISTER_DROPOUT_GRAD_KERNEL_NPU(float)

}  // namespace
}  // namespace oneflow
