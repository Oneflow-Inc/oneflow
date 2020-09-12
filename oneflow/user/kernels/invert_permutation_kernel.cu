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
// #include <cub/cub.cuh>

namespace oneflow {

namespace {

template <typename T>
__global__ void InvertPermutationGpu(const int64_t n, const T *x, T *y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[x[i]] = i; }
}

template <typename T>
void InvertPermutation(DeviceCtx *ctx, const int64_t n, const T *x, T *y) {
  InvertPermutationGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                            ctx->cuda_stream()>>>(n, x, y);
}

template <typename T>
class GpuInvertPermutationKernel final : public user_op::OpKernel {
public:
  GpuInvertPermutationKernel() = default;
  ~GpuInvertPermutationKernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t elem_cnt = in->shape().elem_cnt();
    const T *x_ptr = in->dptr<T>();
    T *y_ptr = out->mut_dptr<T>();

    cudaMemset(y_ptr, -1, elem_cnt * sizeof(T));

    InvertPermutation<T>(ctx->device_ctx(), elem_cnt, x_ptr, y_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_INVERT_PERMUTATION_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("invert_permutation")                                   \
      .SetCreateFn<GpuInvertPermutationKernel<dtype>>()                        \
      .SetIsMatchedHob(                                                        \
          (user_op::HobDeviceTag() == "gpu") &                                 \
          (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_GPU_INVERT_PERMUTATION_KERNEL(int32_t)
REGISTER_GPU_INVERT_PERMUTATION_KERNEL(int64_t)

} //  namespace

} // namespace oneflow