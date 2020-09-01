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
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace {

__global__ void HalfAddByScalarPtrGpu(const int64_t n, const half* x, const half* y, half* z) {
  const half y_value = y[0];
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] + y_value; }
}

}  // namespace

class HalfScalarAddByTensorKernel final : public user_op::OpKernel {
 public:
  HalfScalarAddByTensorKernel() = default;
  ~HalfScalarAddByTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    RUN_CUDA_KERNEL(HalfAddByScalarPtrGpu, ctx->device_ctx(), y->shape().elem_cnt(), 
                    y->shape().elem_cnt(), reinterpret_cast<const half*>(x->dptr<float16>()),
                    reinterpret_cast<const half*>(scalar->dptr<float16>()),
                    reinterpret_cast<half*>(y->mut_dptr<float16>()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


#define REGISTER_SCALAR_ADD_BY_TENSOR_HALF_KERNEL \
  REGISTER_USER_KERNEL("scalar_add_by_tensor")                  \                                        
    .SetCreateFn<HalfScalarAddByTensorKernel>()        \    
    .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)         \
                    & (user_op::HobDataType("x", 0) ==  GetDataType<float16>::value) \
                    & (user_op::HobDataType("y", 0) == GetDataType<float16>::value))       \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });
REGISTER_SCALAR_ADD_BY_TENSOR_HALF_KERNEL

}  // namespace oneflow