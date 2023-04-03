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
#include "oneflow/user/kernels/radix_sort.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ScalarAdd(int32_t elem_cnt, const T* in, const T scalar, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out[i] = in[i] + scalar; };
}

}  // namespace

template<typename T>
class HostScalarAddByTensorKernel final : public user_op::OpKernel {
 public:
  HostScalarAddByTensorKernel() = default;
  ~HostScalarAddByTensorKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const int32_t elem_cnt = x->shape_view().elem_cnt();

    CHECK_EQ(scalar->shape_view().elem_cnt(), 1);

    // val of scalar can be visited because it is host input.
    const T scalar_val = *scalar->dptr<T>();

    ScalarAdd<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(elem_cnt, x->dptr<T>(),
                                                                      scalar_val, y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ARG_SORT_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("host_scalar_add_by_tensor")                                        \
      .SetCreateFn<HostScalarAddByTensorKernel<dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)      \
                       && (user_op::HobDataType("scalar", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_ARG_SORT_KERNEL(float)
REGISTER_CUDA_ARG_SORT_KERNEL(double)
REGISTER_CUDA_ARG_SORT_KERNEL(bool)
REGISTER_CUDA_ARG_SORT_KERNEL(int8_t)
REGISTER_CUDA_ARG_SORT_KERNEL(uint8_t)
REGISTER_CUDA_ARG_SORT_KERNEL(int32_t)
REGISTER_CUDA_ARG_SORT_KERNEL(int64_t)

}  // namespace oneflow
