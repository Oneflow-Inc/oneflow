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
#include "oneflow/user/kernels/roll_kernel_utils.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#include <algorithm>

namespace oneflow {

namespace {

template<typename T>
__global__ void repeat_interleave(const T* in_ptr, const T* cumsum_ptr, T* out_ptr,
                                  const int64_t num) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    T end = cumsum_ptr[i];
    T size = in_ptr[i];
    T start = end - size;
    for (T j = start; j < end; j++) { out_ptr[j] = i; }
  }
}

}  // namespace

template<typename T>
class GpuRepeatInterLeaveKernel final : public user_op::OpKernel {
 public:
  GpuRepeatInterLeaveKernel() = default;
  ~GpuRepeatInterLeaveKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* cumsum = ctx->Tensor4ArgNameAndIndex("cumsum", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t& repeat_num = ctx->Attr<std::int64_t>("repeat_num");
    const T* in_ptr = in->dptr<T>();
    const T* cumsum_ptr = cumsum->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    repeat_interleave<T><<<BlocksNum4ThreadsNum(in->shape_view().At(0)), kCudaThreadsNumPerBlock, 0,
                           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        in_ptr, cumsum_ptr, out_ptr, in->shape_view().At(0));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REPEAT_INTER_LEAVE_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("repeat_interleave")                            \
      .SetCreateFn<GpuRepeatInterLeaveKernel<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_REPEAT_INTER_LEAVE_KERNEL(int32_t);
REGISTER_REPEAT_INTER_LEAVE_KERNEL(int64_t);

}  // namespace oneflow
