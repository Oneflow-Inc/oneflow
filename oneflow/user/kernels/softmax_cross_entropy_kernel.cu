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
#include "oneflow/user/kernels/softmax_cross_entropy_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {
namespace user_op {

namespace {

constexpr int64_t kCrossEntropyGpuBlockSize = 128;

template<typename T>
__global__ void ComputeEntropyGpu(const int64_t num_instances, const int64_t num_classes,
                                  const T* x, const T* labels, T* y) {
  typedef cub::BlockReduce<T, kCrossEntropyGpuBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < num_instances; row += gridDim.x) {
    const int row_offset = row * num_classes;
    const T* in_row = x + row_offset;
    const T* label_row = labels + row_offset;
    T result = 0;
    for (int col = tid; col < num_classes; col += kCrossEntropyGpuBlockSize) {
      T label = label_row[col];
      T prob = in_row[col];
      result += -label * SafeLog(prob);
    }
    __syncthreads();
    T row_reduce_result = BlockReduce(temp_storage).Reduce(result, cub::Sum());
    if (0 == tid) { y[row] = row_reduce_result; }
  }
}

__global__ void ComputeEntropyGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                      const half* x, const half* labels, half* y) {
  typedef cub::BlockReduce<float, kCrossEntropyGpuBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < num_instances; row += gridDim.x) {
    const int row_offset = row * num_classes;
    const half* in_row = x + row_offset;
    const half* label_row = labels + row_offset;
    float result = 0;
    for (int col = tid; col < num_classes; col += kCrossEntropyGpuBlockSize) {
      float label = __half2float(label_row[col]);
      float prob = __half2float(in_row[col]);
      result += -label * SafeLog(prob);
    }
    __syncthreads();
    float row_reduce_result = BlockReduce(temp_storage).Reduce(result, cub::Sum());
    if (0 == tid) { y[row] = __float2half(row_reduce_result); }
  }
}

template<typename T>
__global__ void ComputeDiffWithSoftmaxGpu(const int64_t elem_cnt, const int64_t num_classes,
                                          const T* prob, const T* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    dx[i] = dy[row_id] * (prob[i] - labels[i]);
  }
}

__global__ void ComputeDiffWithSoftmaxGpuHalf(const int64_t elem_cnt, const int64_t num_classes,
                                              const half* prob, const half* labels, const half* dy,
                                              half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    dx[i] = __hmul(dy[row_id], __hsub(prob[i], labels[i]));
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

}  // namespace

int GetCrossEntropyNumBlocks(const int num_instances) {
  return std::min(static_cast<int>(num_instances), kCudaMaxBlocksNum);
}

int GetCrossEntropyBlockSize() { return kCrossEntropyGpuBlockSize; }

template<typename T>
struct CrossEntropyKernelUtil<DeviceType::kGPU, T> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const T* x, const T* labels, T* y) {
    cudaMemset(y, 0, sizeof(T) * num_instances);
    ComputeEntropyGpu<<<GetCrossEntropyNumBlocks(num_instances), GetCrossEntropyBlockSize(), 0,
                        ctx->cuda_stream()>>>(num_instances, num_classes, x, labels, y);
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const T* prob, const T* labels,
                                     const T* dy, T* dx) {
    ComputeDiffWithSoftmaxGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx->cuda_stream()>>>(elem_cnt, num_classes, prob, labels, dy, dx);
  }
};

template<>
struct CrossEntropyKernelUtil<DeviceType::kGPU, float16> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const float16* x, const float16* labels, float16* y) {
    cudaMemset(y, 0, sizeof(float16) * num_instances);
    ComputeEntropyGpuHalf<<<GetCrossEntropyNumBlocks(num_instances), GetCrossEntropyBlockSize(), 0,
                            ctx->cuda_stream()>>>(
        num_instances, num_classes, reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(labels), reinterpret_cast<half*>(y));
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const float16* prob,
                                     const float16* labels, const float16* dy, float16* dx) {
    ComputeDiffWithSoftmaxGpuHalf<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                    ctx->cuda_stream()>>>(
        elem_cnt, num_classes, reinterpret_cast<const half*>(prob),
        reinterpret_cast<const half*>(labels), reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
