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

template<typename CondT>
__global__ void NaiveHalfWhere(const int64_t elem_cnt, const CondT* cond, const half* lhs,
                               const half* rhs, half* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out[i] = static_cast<bool>(cond[i]) ? lhs[i] : rhs[i];
  }
}

}  // namespace

template<typename CondT>
class HalfWhereUserKernel final : public user_op::OpKernel {
 public:
  HalfWhereUserKernel() = default;
  ~HalfWhereUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    RUN_CUDA_KERNEL((NaiveHalfWhere<CondT>), ctx->device_ctx(), out->shape().elem_cnt(), 
                    out->shape().elem_cnt(), cond->dptr<CondT>(),
                    reinterpret_cast<const half*>(x->dptr<float16>()),
                    reinterpret_cast<const half*>(y->dptr<float16>()),
                    reinterpret_cast<half*>(out->mut_dptr<float16>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WHERE_HALF_KERNEL(ctype) \
  REGISTER_USER_KERNEL("where")                  \                                        
    .SetCreateFn<HalfWhereUserKernel<ctype>>()        \    
    .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)         \
                    & (user_op::HobDataType("condition", 0) ==  GetDataType<ctype>::value) \
                    & (user_op::HobDataType("out", 0) == GetDataType<float16>::value));

REGISTER_WHERE_HALF_KERNEL(int8_t)
REGISTER_WHERE_HALF_KERNEL(int32_t)
REGISTER_WHERE_HALF_KERNEL(int64_t)

}  // namespace oneflow