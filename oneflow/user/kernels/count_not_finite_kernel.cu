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
#include <cub/cub.cuh>
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T, int32_t N>
struct Param {
  const T* x[N];
  int64_t x_elem_cnt[N];
  int64_t* y;
  int64_t num_x;
};

using CuInt64T = unsigned long long int;

__device__ __inline__ int64_t AtomicAdd(int64_t* address, int64_t val) {
  static_assert(sizeof(int64_t) == sizeof(CuInt64T), "size error");
  return static_cast<int64_t>(
      atomicAdd(reinterpret_cast<CuInt64T*>(address), static_cast<CuInt64T>(val)));
}

template<typename T>
__global__ void CountNotFiniteGpu(const int64_t n, const T* x, int64_t* y) {
  typedef cub::BlockReduce<int64_t, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  int64_t thread_count = 0;
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (!isfinite(x[i])) { thread_count += 1; }
  }
  __syncthreads();
  int64_t block_count_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_count, cub::Sum());
  if (threadIdx.x == 0) { AtomicAdd(y, block_count_sum); }
}

template<typename T, int32_t N>
__global__ void MultiCountNotFiniteGpu(Param<T, N> param) {
  typedef cub::BlockReduce<int64_t, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  int64_t thread_count = 0;
  for (int32_t k = 0; k < param.num_x; ++k) {
    CUDA_1D_KERNEL_LOOP(i, param.x_elem_cnt[k]) {
      if (!isfinite(param.x[k][i])) { thread_count += 1; }
    }
  }
  __syncthreads();
  int64_t block_count_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_count, cub::Sum());
  if (threadIdx.x == 0) { AtomicAdd(param.y, block_count_sum); }
}

constexpr int64_t kCountNotFiniteNumBlocks = 512;

int GetCountNotFiniteNumBlocks(const int64_t elem_cnt) {
  return std::min((elem_cnt + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock,
                  kCountNotFiniteNumBlocks);
}

}  // namespace

template<typename T>
class CountNotFiniteGpuKernel final : public user_op::OpKernel {
 public:
  CountNotFiniteGpuKernel() = default;
  ~CountNotFiniteGpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t elem_cnt = x->shape().elem_cnt();
    Memset<DeviceType::kGPU>(ctx->device_ctx(), y->mut_dptr<int64_t>(), 0,
                             y->shape().elem_cnt() * sizeof(int64_t));
    CountNotFiniteGpu<T>
        <<<GetCountNotFiniteNumBlocks(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->device_ctx()->cuda_stream()>>>(elem_cnt, x->dptr<T>(), y->mut_dptr<int64_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COUNT_NOT_FINITE_GPU_KERNEL(dtype)       \
  REGISTER_USER_KERNEL("count_not_finite")                \
      .SetCreateFn<CountNotFiniteGpuKernel<dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_COUNT_NOT_FINITE_GPU_KERNEL(float)
REGISTER_COUNT_NOT_FINITE_GPU_KERNEL(double)

template<typename T>
class MultiCountNotFiniteGpuKernel final : public user_op::OpKernel {
 public:
  MultiCountNotFiniteGpuKernel() = default;
  ~MultiCountNotFiniteGpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    Param<T, 128> para;
    Memset<DeviceType::kGPU>(ctx->device_ctx(), y->mut_dptr<int64_t>(), 0,
                             y->shape().elem_cnt() * sizeof(int64_t));
    para.y = y->mut_dptr<int64_t>();

    int64_t remain_size = ctx->inputs().size();
    int64_t input_id = 0;
    while (remain_size > 0) {
      int64_t num_x = 0;
      if (remain_size > 128) {
        remain_size -= 128;
        para.num_x = 128;
      } else {
        para.num_x = remain_size;
        remain_size = 0;
      }
      int64_t max_elem_cnt = 0;
      for (int32_t i = 0; i < para.num_x; ++i) {
        const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", input_id);
        input_id++;
        para.x[i] = x->dptr<T>();
        para.x_elem_cnt[i] = x->shape().elem_cnt();
        max_elem_cnt = std::max(max_elem_cnt, x->shape().elem_cnt());
      }
      MultiCountNotFiniteGpu<T, 128>
          <<<GetCountNotFiniteNumBlocks(max_elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->device_ctx()->cuda_stream()>>>(para);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MULTI_COUNT_NOT_FINITE_GPU_KERNEL(dtype) \
  REGISTER_USER_KERNEL("multi_count_not_finite")          \
      .SetCreateFn<MultiCountNotFiniteGpuKernel<dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MULTI_COUNT_NOT_FINITE_GPU_KERNEL(float)
REGISTER_MULTI_COUNT_NOT_FINITE_GPU_KERNEL(double)

}  // namespace oneflow
