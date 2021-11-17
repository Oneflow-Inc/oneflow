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
#include <cub/cub.cuh>
#include "oneflow/user/kernels/loss_kernel_util.h"

namespace oneflow {
namespace user_op {
namespace loss {

template<typename T>
__global__ void ApplyLossReductionImpl(int64_t elem_cnt, double inv_elem_cnt, const T* tmp_out,
                                       T* out, bool is_reduce_mean) {
  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  T thread_sum = static_cast<T>(0);
  for (int i = threadIdx.x; i < elem_cnt; i += kCudaThreadsNumPerBlock) {
    thread_sum += tmp_out[i];
  }
  __syncthreads();
  T block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
  if (threadIdx.x == 0) {
    T out_val = block_sum;
    if (is_reduce_mean) { out_val *= static_cast<T>(inv_elem_cnt); }
    *out = out_val;
  }
}

template<>
__global__ void ApplyLossReductionImpl<half>(int64_t elem_cnt, double inv_elem_cnt,
                                             const half* tmp_out, half* out, bool is_reduce_mean) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  typedef cub::BlockReduce<half, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage cub_reduce_tmp_storage;
  half thread_sum = __float2half(0.0);
  for (int i = threadIdx.x; i < elem_cnt; i += kCudaThreadsNumPerBlock) {
    thread_sum = __hadd(thread_sum, tmp_out[i]);
  }
  __syncthreads();
  half block_sum = BlockReduce(cub_reduce_tmp_storage).Reduce(thread_sum, cub::Sum());
  if (threadIdx.x == 0) {
    half out_val = block_sum;
    if (is_reduce_mean) {
      out_val = __float2half(__half2float(*out) * static_cast<float>(inv_elem_cnt));
    }
    *out = out_val;
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<DeviceType device_type, typename T>
RETURN_VOID_IF_GPU(device_type)
ApplyLossReductionIfNeed(DeviceCtx* ctx, int64_t elem_cnt, const T* tmp_out, T* out,
                         const ReductionType reduction_type) {
  if (reduction_type == ReductionType::kNone) { return; }
  if ((reduction_type != ReductionType::kMean) && (reduction_type != ReductionType::kSum)) {
    UNIMPLEMENTED();
    return;
  }
  ApplyLossReductionImpl<<<1, kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      elem_cnt, static_cast<double>(1.0 / elem_cnt), tmp_out, out,
      reduction_type == ReductionType::kMean);
}

#define SPECIALIZE_APPLY_LOSS_REDUCTION(device_type, dtype)                              \
  template RETURN_VOID_IF_GPU(device_type) ApplyLossReductionIfNeed<device_type, dtype>( \
      DeviceCtx * ctx, int64_t elem_cnt, const dtype* tmp_out, dtype* out,               \
      const ReductionType reduction_type);

SPECIALIZE_APPLY_LOSS_REDUCTION(DeviceType::kGPU, half)
SPECIALIZE_APPLY_LOSS_REDUCTION(DeviceType::kGPU, float)
SPECIALIZE_APPLY_LOSS_REDUCTION(DeviceType::kGPU, double)

}  // namespace loss
}  // namespace user_op
}  // namespace oneflow
