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
#include <glog/logging.h>
#include "oneflow/user/kernels/variance_kernel_util.h"
#include "oneflow/core/cuda/layer_norm.cuh"

namespace oneflow {
namespace user_op {

namespace {
template<typename T>
__inline__ __device__ T Nan();

template<>
__inline__ __device__ float Nan<float>() {
  return CUDART_NAN_F;
}

template<>
__inline__ __device__ double Nan<double>() {
  return CUDART_NAN;
}
}  // namespace

template<typename T>
__global__ void ComputeVarUsingWelfordWrapper(const T* in_ptr, T* out_ptr, const VarParam var_param,
                                              bool is_nan) {
  if (is_nan) {
    CUDA_1D_KERNEL_LOOP(i, var_param.parallel_num) { out_ptr[i] = Nan<T>(); }
  } else {
    CUDA_1D_KERNEL_LOOP(i, var_param.parallel_num) {
      const size_t input_offset = LinearIndex2Offset(
          i, var_param.dim_size_in_caxis, var_param.stride_in_caxis, var_param.caxis_size);
      ComputeVarUsingWelford(&in_ptr[input_offset], &out_ptr[i], var_param);
    }
  }
}

namespace {
template<typename T>
inline __device__ void WelfordReduce(const T* in_ptr, T* mean, T* m2, T* count,
                                     const size_t total_elem_cnt, const size_t start,
                                     const size_t step) {
  T old_mean = 0.0;
  for (size_t i = start; i < total_elem_cnt; i += step) {
    ++(*count);
    old_mean = *mean;
    *mean += (in_ptr[i] - *mean) / *count;
    *m2 += (in_ptr[i] - *mean) * (in_ptr[i] - old_mean);
  }
}

template<typename T>
inline __device__ void WelfordCombine(const T* b_mean, const T* b_m2, const T* b_count, T* mean,
                                      T* m2, T* count, const size_t total_elem_cnt,
                                      const size_t start, const size_t step) {
  for (size_t i = start; i < total_elem_cnt; i += step) {
    cuda::layer_norm::WelfordCombine(b_mean[i], b_m2[i], b_count[i], mean, m2, count);
  }
}
__device__ int32_t done_block_count = 0;
}  // namespace

template<typename T>
__global__ void ComputeVarScalarOut(const T* in_ptr, T* out_ptr, T* tmp_buffer_ptr,
                                    const VarParam var_param) {
  if (var_param.elem_cnt == 1 && var_param.unbiased == true) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { *out_ptr = Nan<T>(); }
    return;
  }
  const size_t elems_per_block = var_param.elem_cnt / gridDim.x;
  const size_t elems_per_thread = elems_per_block / blockDim.x;
  // tail element number in block
  size_t tail_elems = elems_per_block % blockDim.x;

  T thread_mean = 0.0;
  T thread_m2 = 0.0;
  T thread_count = 0.0;
  // every thread deal it's elems
  if (elems_per_thread > 0) {
    const size_t block_offset = blockIdx.x * elems_per_block;
    WelfordReduce(&in_ptr[block_offset], &thread_mean, &thread_m2, &thread_count,
                  elems_per_block - tail_elems, threadIdx.x, blockDim.x);
  }
  // thread 0 of last block handles tail element between blocks
  if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
    tail_elems += var_param.elem_cnt % gridDim.x;
  }
  // thread 0 deal tail elems
  if (tail_elems != 0 && threadIdx.x == 0) {
    const size_t tail_offset = blockIdx.x * elems_per_block + blockDim.x * elems_per_thread;
    WelfordReduce(&in_ptr[tail_offset], &thread_mean, &thread_m2, &thread_count, tail_elems,
                  /*tail start=*/0, /*step=*/1);
  }

  T block_mean = 0;
  T block_m2 = 0;
  T block_count = 0;
  cuda::layer_norm::WelfordBlockAllReduce<T>(thread_mean, thread_m2, thread_count, &block_mean,
                                             &block_m2, &block_count);

  if (gridDim.x == 1) {
    if (threadIdx.x == 0) {
      *out_ptr =
          cuda::layer_norm::Div(block_m2, (var_param.unbiased ? block_count - 1 : block_count));
    }
    return;
  }

  T* tmp_mean_ptr = tmp_buffer_ptr;
  T* tmp_m2_ptr = &tmp_mean_ptr[gridDim.x];
  T* tmp_count_ptr = &tmp_m2_ptr[gridDim.x];
  if (threadIdx.x == 0) {
    tmp_mean_ptr[blockIdx.x] = block_mean;
    tmp_m2_ptr[blockIdx.x] = block_m2;
    tmp_count_ptr[blockIdx.x] = block_count;
  }
  __shared__ bool is_last_block;
  if (threadIdx.x == 0) { is_last_block = atomicAdd(&done_block_count, 1) == gridDim.x - 1; }
  __syncthreads();
  if (is_last_block) {
    T last_block_thread_mean = 0;
    T last_block_thread_m2 = 0;
    T last_block_thread_count = 0;
    const size_t welforddatas_per_thread = gridDim.x / blockDim.x;
    const size_t tail_welforddatas = gridDim.x % blockDim.x;

    if (welforddatas_per_thread > 0) {
      WelfordCombine(tmp_mean_ptr, tmp_m2_ptr, tmp_count_ptr, &last_block_thread_mean,
                     &last_block_thread_m2, &last_block_thread_count, gridDim.x - tail_welforddatas,
                     threadIdx.x, blockDim.x);
    }
    // thread 0 deal tail welford data
    if (tail_welforddatas != 0 && threadIdx.x == 0) {
      const size_t last_block_tail_offset = blockDim.x * welforddatas_per_thread;
      WelfordCombine(&tmp_mean_ptr[last_block_tail_offset], &tmp_m2_ptr[last_block_tail_offset],
                     &tmp_count_ptr[last_block_tail_offset], &last_block_thread_mean,
                     &last_block_thread_m2, &last_block_thread_count, tail_welforddatas,
                     /*tail start=*/0, /*step=*/1);
    }
    T final_mean = 0;
    T final_m2 = 0;
    T final_count = 0;
    cuda::layer_norm::WelfordBlockAllReduce<T>(last_block_thread_mean, last_block_thread_m2,
                                               last_block_thread_count, &final_mean, &final_m2,
                                               &final_count);
    if (threadIdx.x == 0) {
      *out_ptr =
          cuda::layer_norm::Div(final_m2, (var_param.unbiased ? final_count - 1 : final_count));
      done_block_count = 0;
    }
  }
}

template<typename T>
struct VarFunctor<DeviceType::kCUDA, T> final {
  void operator()(ep::Stream* stream, const T* in_ptr, T* out_ptr, T* tmp_buffer_ptr,
                  const VarParam var_param) {
    int grid_dim = 0;
    int block_dim = 0;
    SetGridDimAndBlockDim(var_param.elem_cnt, &grid_dim, &block_dim);
    if (var_param.parallel_num == 1) {
      ComputeVarScalarOut<T>
          <<<grid_dim, block_dim, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              in_ptr, out_ptr, tmp_buffer_ptr, var_param);
    } else {
      // if var_param.parallel_num is 0, do nothing, return 0-size tensor
      if (var_param.parallel_num == 0) { return; }
      RUN_CUDA_KERNEL(ComputeVarUsingWelfordWrapper<T>, stream, var_param.parallel_num, in_ptr,
                      out_ptr, var_param, IsNanOut(var_param));
    }
  }
};

template struct VarFunctor<DeviceType::kCUDA, float>;
template struct VarFunctor<DeviceType::kCUDA, double>;
}  // namespace user_op
}  // namespace oneflow
