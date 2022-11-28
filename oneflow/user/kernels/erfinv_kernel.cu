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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

template<typename T>
struct ErfInvFunctor {
  OF_DEVICE_FUNC ErfInvFunctor() {}
  OF_DEVICE_FUNC T operator()(T x) const { return erfinv(x); }
};

template<typename T>
class GpuErfinvKernel final : public user_op::OpKernel {
 public:
  GpuErfinvKernel() = default;
  ~GpuErfinvKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    OF_CUDA_CHECK(cuda::elementwise::Unary(ErfInvFunctor<T>(), elem_cnt, y->mut_dptr<T>(),
                                           x->dptr<T>(),
                                           ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ERFINV_KERNEL(dtype)                                                      \
  REGISTER_USER_KERNEL("erfinv")                                                                \
      .SetCreateFn<GpuErfinvKernel<dtype>>()                                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value))          \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_CUDA_ERFINV_KERNEL(float)
REGISTER_CUDA_ERFINV_KERNEL(double)

}  // namespace oneflow
