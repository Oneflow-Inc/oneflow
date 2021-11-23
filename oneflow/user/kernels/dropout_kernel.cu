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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

/*
curand_uniform4 interval is (0, 1.0]
*/

constexpr int32_t kVecSize = 4;
constexpr int32_t kBlockSize = 256;
constexpr int32_t PackDoubleSize = 2;
constexpr int32_t PackFloatSize = 4;
constexpr int32_t PackHalfSize = 4;
constexpr int32_t PackHalf2Size = 2;

using H2PackType = typename std::aligned_storage<4 * sizeof(half), 4 * sizeof(half)>::type;
union H2Pack {
  H2PackType storage;
  half2 h2[2];
};

union RandPack4 {
  float4 storage;
  float elem[4];
};

#define RETURN_VOID_IF_HALF typename std::enable_if_t<std::is_same<T, half>::value, void>
#define RETURN_VOID_IF_FLOAT typename std::enable_if_t<std::is_same<T, float>::value, void>
#define RETURN_VOID_IF_DOUBLE typename std::enable_if_t<std::is_same<T, double>::value, void>

template<typename T, bool tail>
__global__ RETURN_VOID_IF_FLOAT MaskAndScaleGpu(uint64_t seed,
                                                one::CUDAGeneratorState* cuda_gen_state,
                                                uint64_t counter_offset, const int64_t elem_cnt,
                                                float rate, float scale, int64_t n_tail, const T* x,
                                                int8_t* mask, T* y, const T* tail_x,
                                                int8_t* tail_mask, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadT =
      typename std::aligned_storage<sizeof(T) * PackFloatSize, sizeof(T) * PackFloatSize>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * PackFloatSize,
                                              sizeof(int8_t) * PackFloatSize>::type;
  RandPack4 rand_uniform_pack4;

  for (int64_t linear_index = global_thread_id * PackFloatSize; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * PackFloatSize) {
    rand_uniform_pack4.storage = curand_uniform4(&state);

    const LoadT* x_load = reinterpret_cast<const LoadT*>(x + linear_index);
    cuda::elementwise::Pack<T, PackFloatSize> x_vec;
    x_vec.storage = *x_load;

    int8_t mask_vec[PackFloatSize];
    T y_vec[PackFloatSize];
#pragma unroll
    for (int i = 0; i < PackFloatSize; i++) {
      rand_uniform_pack4.elem[i] = rand_uniform_pack4.elem[i] >= rate;
      mask_vec[i] = rand_uniform_pack4.elem[i];
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale;
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }

  if (tail && global_thread_id < n_tail && global_thread_id == 0) {
    rand_uniform_pack4.storage = curand_uniform4(&state);
#pragma unroll
    for (int i = 0; i < n_tail; i++) {
      rand_uniform_pack4.elem[i] = rand_uniform_pack4.elem[i] >= rate;
      tail_mask[i] = rand_uniform_pack4.elem[i];
      tail_y[i] = tail_x[i] * rand_uniform_pack4.elem[i] * scale;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;               // reset counter to zero
      cuda_gen_state->dev_offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, bool tail>
__global__ RETURN_VOID_IF_FLOAT MaskAndScaleAddGpu(
    uint64_t seed, one::CUDAGeneratorState* cuda_gen_state, uint64_t counter_offset,
    const int64_t elem_cnt, float rate, float scale, int64_t n_tail, const T* x, int8_t* mask,
    const T* addend, T* y, const T* tail_x, int8_t* tail_mask, const T* tail_addend, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadT =
      typename std::aligned_storage<sizeof(T) * PackFloatSize, sizeof(T) * PackFloatSize>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * PackFloatSize,
                                              sizeof(int8_t) * PackFloatSize>::type;

  RandPack4 rand_uniform_pack4;
  for (int64_t linear_index = global_thread_id * PackFloatSize; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * PackFloatSize) {
    rand_uniform_pack4.storage = curand_uniform4(&state);

    const LoadT* x_load = reinterpret_cast<const LoadT*>(x + linear_index);
    cuda::elementwise::Pack<T, PackFloatSize> x_vec;
    x_vec.storage = *x_load;

    const LoadT* addend_load = reinterpret_cast<const LoadT*>(&addend[linear_index]);
    cuda::elementwise::Pack<T, PackFloatSize> addend_vec;
    addend_vec.storage = *addend_load;

    int8_t mask_vec[PackFloatSize];
    T y_vec[PackFloatSize];
#pragma unroll
    for (int i = 0; i < PackFloatSize; i++) {
      mask_vec[i] = rand_uniform_pack4.elem[i];
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale + addend_vec.elem[i];
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }

  if (tail && global_thread_id < n_tail && global_thread_id == 0) {
    rand_uniform_pack4.storage = curand_uniform4(&state);

#pragma unroll
    for (int i = 0; i < n_tail; i++) {
      tail_mask[i] = rand_uniform_pack4.elem[i];
      tail_y[i] = tail_x[i] * rand_uniform_pack4.elem[i] * scale + tail_addend[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;               // reset counter to zero
      cuda_gen_state->dev_offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, bool tail>
__global__ RETURN_VOID_IF_HALF MaskAndScaleGpu(uint64_t seed,
                                               one::CUDAGeneratorState* cuda_gen_state,
                                               uint64_t counter_offset, const int64_t elem_cnt,
                                               float rate, float scale, int64_t n_tail, const T* x,
                                               int8_t* mask, T* y, const T* tail_x,
                                               int8_t* tail_mask, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadT =
      typename std::aligned_storage<sizeof(half) * PackHalfSize, sizeof(half) * PackHalfSize>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * PackHalfSize,
                                              sizeof(int8_t) * PackHalfSize>::type;

  RandPack4 rand_uniform_pack4;
  half2 h2_scale = __float2half2_rn(scale);
  for (int64_t linear_index = global_thread_id * PackHalfSize; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * PackHalfSize) {
    rand_uniform_pack4.storage = curand_uniform4(&state);
    const LoadT* x_load = reinterpret_cast<const LoadT*>(x + linear_index);
    H2Pack x_vec{};
    x_vec.storage = *x_load;

    int8_t mask_vec[PackHalfSize];
    half2 y_vec[PackHalf2Size];
    half2 one_or_zero_h2[PackHalf2Size];

    mask_vec[0] = rand_uniform_pack4.elem[0] > rate;
    one_or_zero_h2[0].x = mask_vec[0];
    mask_vec[1] = rand_uniform_pack4.elem[1] > rate;
    one_or_zero_h2[0].y = mask_vec[1];
    y_vec[0] = __hmul2(__hmul2(x_vec.h2[0], one_or_zero_h2[0]), h2_scale);

    mask_vec[2] = rand_uniform_pack4.elem[2] > rate;
    one_or_zero_h2[1].x = mask_vec[2];
    mask_vec[3] = rand_uniform_pack4.elem[3] > rate;
    one_or_zero_h2[1].y = mask_vec[3];
    y_vec[1] = __hmul2(__hmul2(x_vec.h2[1], one_or_zero_h2[1]), h2_scale);

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }

  if (tail && global_thread_id < n_tail && global_thread_id == 0) {
    half half_scale = __float2half_rn(scale);
    rand_uniform_pack4.storage = curand_uniform4(&state);
#pragma unroll
    for (int i = 0; i < n_tail; i++) {
      rand_uniform_pack4.elem[i] = rand_uniform_pack4.elem[i] > rate;
      tail_mask[i] = rand_uniform_pack4.elem[i];
      tail_y[i] = tail_x[i] * static_cast<half>(rand_uniform_pack4.elem[i]) * half_scale;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;               // reset counter to zero
      cuda_gen_state->dev_offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, bool tail>
__global__ RETURN_VOID_IF_HALF MaskAndScaleAddGpu(
    uint64_t seed, one::CUDAGeneratorState* cuda_gen_state, uint64_t counter_offset,
    const int64_t elem_cnt, float rate, float scale, int64_t n_tail, const T* x, int8_t* mask,
    const T* addend, T* y, const T* tail_x, int8_t* tail_mask, const T* tail_addend, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadT =
      typename std::aligned_storage<sizeof(half) * PackHalfSize, sizeof(half) * PackHalfSize>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * PackHalfSize,
                                              sizeof(int8_t) * PackHalfSize>::type;

  RandPack4 rand_uniform_pack4;
  half2 h2_scale = __float2half2_rn(scale);
  for (int64_t linear_index = global_thread_id * PackHalfSize; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * PackHalfSize) {
    rand_uniform_pack4.storage = curand_uniform4(&state);
    const LoadT* x_load = reinterpret_cast<const LoadT*>(x + linear_index);
    H2Pack x_vec{};
    x_vec.storage = *x_load;

    const LoadT* addend_load = reinterpret_cast<const LoadT*>(&addend[linear_index]);
    H2Pack addend_vec{};
    addend_vec.storage = *addend_load;

    int8_t mask_vec[PackHalfSize];
    half2 y_vec[PackHalf2Size];
    half2 one_or_zero_h2[PackHalf2Size];

    mask_vec[0] = rand_uniform_pack4.elem[0] > rate;
    one_or_zero_h2[0].x = mask_vec[0];
    mask_vec[1] = rand_uniform_pack4.elem[1] > rate;
    one_or_zero_h2[0].y = mask_vec[1];
    y_vec[0] =
        __hadd2(__hmul2(__hmul2(x_vec.h2[0], one_or_zero_h2[0]), h2_scale), addend_vec.h2[0]);

    mask_vec[2] = rand_uniform_pack4.elem[2] > rate;
    one_or_zero_h2[1].x = mask_vec[2];
    mask_vec[3] = rand_uniform_pack4.elem[3] > rate;
    one_or_zero_h2[1].y = mask_vec[3];
    y_vec[1] =
        __hadd2(__hmul2(__hmul2(x_vec.h2[1], one_or_zero_h2[1]), h2_scale), addend_vec.h2[0]);

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }

  if (tail && global_thread_id < n_tail && global_thread_id == 0) {
    half half_scale = __float2half_rn(scale);
    rand_uniform_pack4.storage = curand_uniform4(&state);
#pragma unroll
    for (int i = 0; i < n_tail; i++) {
      rand_uniform_pack4.elem[i] = rand_uniform_pack4.elem[i] > rate;
      tail_mask[i] = rand_uniform_pack4.elem[i];
      tail_y[i] =
          tail_x[i] * static_cast<half>(rand_uniform_pack4.elem[i]) * half_scale + addend[i];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;               // reset counter to zero
      cuda_gen_state->dev_offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, bool tail>
__global__ RETURN_VOID_IF_DOUBLE MaskAndScaleGpu(uint64_t seed,
                                                 one::CUDAGeneratorState* cuda_gen_state,
                                                 uint64_t counter_offset, const int64_t elem_cnt,
                                                 float rate, float scale, int64_t n_tail,
                                                 const T* x, int8_t* mask, T* y, const T* tail_x,
                                                 int8_t* tail_mask, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(double) * PackDoubleSize,
                                              sizeof(double) * PackDoubleSize>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * PackDoubleSize,
                                              sizeof(int8_t) * PackDoubleSize>::type;
  RandPack4 rand_uniform_pack4;
  bool grid_loop_rand_state = 0;

  for (int64_t linear_index = global_thread_id * PackDoubleSize; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * PackDoubleSize) {
    if (grid_loop_rand_state == 0) {
      rand_uniform_pack4.storage = curand_uniform4(&state);
    } else {
      // Use the last two random numbers we generated in previous iteration.
      rand_uniform_pack4.elem[0] = rand_uniform_pack4.elem[2];
      rand_uniform_pack4.elem[1] = rand_uniform_pack4.elem[3];
      grid_loop_rand_state ^= 1;
    }
    const LoadT* x_load = reinterpret_cast<const LoadT*>(x + linear_index);
    cuda::elementwise::Pack<double, PackDoubleSize> x_vec;
    x_vec.storage = *x_load;

    int8_t mask_vec[PackDoubleSize];
    double y_vec[PackDoubleSize];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      mask_vec[i] = rand_uniform_pack4.elem[i];
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale;
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }

  if (tail && global_thread_id < n_tail && global_thread_id == 0) {
    float single_rand_uniform = curand_uniform(&state);
    single_rand_uniform = single_rand_uniform > rate;
#pragma unroll
    for (int i = 0; i < n_tail; i++) {
      tail_mask[i] = single_rand_uniform;
      tail_y[i] = tail_x[i] * single_rand_uniform * scale;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;               // reset counter to zero
      cuda_gen_state->dev_offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<typename T, bool tail>
__global__ RETURN_VOID_IF_DOUBLE MaskAndScaleAddGpu(
    uint64_t seed, one::CUDAGeneratorState* cuda_gen_state, uint64_t counter_offset,
    const int64_t elem_cnt, float rate, float scale, int64_t n_tail, const T* x, int8_t* mask,
    const T* addend, T* y, const T* tail_x, int8_t* tail_mask, const T* tail_addend, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadT = typename std::aligned_storage<sizeof(double) * PackDoubleSize,
                                              sizeof(double) * PackDoubleSize>::type;
  using MaskT = typename std::aligned_storage<sizeof(int8_t) * PackDoubleSize,
                                              sizeof(int8_t) * PackDoubleSize>::type;

  RandPack4 rand_uniform_pack4;
  bool grid_loop_rand_state = 0;

  for (int64_t linear_index = global_thread_id * PackDoubleSize; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * PackDoubleSize) {
    if (grid_loop_rand_state == 0) {
      rand_uniform_pack4.storage = curand_uniform4(&state);
    } else {
      // Use the last two random numbers we generated in previous iteration.
      rand_uniform_pack4.elem[0] = rand_uniform_pack4.elem[2];
      rand_uniform_pack4.elem[1] = rand_uniform_pack4.elem[3];
      grid_loop_rand_state ^= 1;
    }
    const LoadT* x_load = reinterpret_cast<const LoadT*>(x + linear_index);
    cuda::elementwise::Pack<double, PackDoubleSize> x_vec;
    x_vec.storage = *x_load;

    const LoadT* addend_load = reinterpret_cast<const LoadT*>(&addend[linear_index]);
    cuda::elementwise::Pack<double, PackDoubleSize> addend_vec;
    addend_vec.storage = *addend_load;

    int8_t mask_vec[PackDoubleSize];
    double y_vec[PackDoubleSize];
#pragma unroll
    for (int i = 0; i < PackDoubleSize; i++) {
      rand_uniform_pack4.elem[i] = rand_uniform_pack4.elem[i] > rate;
      mask_vec[i] = rand_uniform_pack4.elem[i];
      y_vec[i] = x_vec.elem[i] * mask_vec[i] * scale + addend_vec.elem[i];
    }

    *(reinterpret_cast<LoadT*>(y + linear_index)) = *reinterpret_cast<LoadT*>(y_vec);
    *(reinterpret_cast<MaskT*>(mask + linear_index)) = *reinterpret_cast<MaskT*>(mask_vec);
  }
  if (tail && global_thread_id < n_tail && global_thread_id == 0) {
    float single_rand_uniform = curand_uniform(&state);
    single_rand_uniform = single_rand_uniform > rate;
#pragma unroll
    for (int i = 0; i < n_tail; i++) {
      tail_mask[i] = single_rand_uniform;
      tail_y[i] = tail_x[i] * single_rand_uniform * scale + addend[i];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;               // reset counter to zero
      cuda_gen_state->dev_offset += counter_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<int pack_size>
unsigned int ComputeGridSize(const int32_t block_size, const int64_t elem_cnt) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
  grid_size = std::min((unsigned int)prop.multiProcessorCount * blocks_per_sm, grid_size);
  return grid_size;
}

template<typename T>
void MaskAndScale(DeviceCtx* ctx, uint64_t seed, one::CUDAGeneratorState* cuda_gen_state,
                  const int64_t elem_cnt, float rate, float scale, const T* x, int8_t* mask, T* y) {
  unsigned int grid_size = ComputeGridSize<4>(kBlockSize, elem_cnt);
  constexpr int pack_size = cuda::elementwise::PackSize<T>();
  const int64_t pack_num = elem_cnt / pack_size;
  const int64_t tail_offset = pack_num * pack_size;
  const int64_t n_tail = pack_num - tail_offset;
  const bool tail = n_tail > 0 ? true : false;
  uint64_t counter_offset = 0;
  if (tail) {
    // If tail, we need generate randnum one more time, so here we add another `1`.
    counter_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1 + 1) * kVecSize;
    MaskAndScaleGpu<T, true><<<grid_size, kBlockSize, 0, ctx->cuda_stream()>>>(
        seed, cuda_gen_state, counter_offset, elem_cnt, rate, scale, n_tail, x, mask, y,
        (x + tail_offset), (mask + tail_offset), (y + tail_offset));
  } else {
    counter_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize;
    MaskAndScaleGpu<T, false><<<grid_size, kBlockSize, 0, ctx->cuda_stream()>>>(
        seed, cuda_gen_state, counter_offset, elem_cnt, rate, scale, n_tail, x, mask, y,
        (x + tail_offset), (mask + tail_offset), (y + tail_offset));
  }
}

template<typename T>
void MaskAndScaleAdd(DeviceCtx* ctx, uint64_t seed, one::CUDAGeneratorState* cuda_gen_state,
                     const int64_t elem_cnt, float rate, float scale, const T* x, int8_t* mask,
                     const T* addend, T* y) {
  unsigned int grid_size = ComputeGridSize<4>(kBlockSize, elem_cnt);
  constexpr int pack_size = cuda::elementwise::PackSize<T>();
  const int64_t pack_num = elem_cnt / pack_size;
  const int64_t tail_offset = pack_num * pack_size;
  const int64_t n_tail = pack_num - tail_offset;
  const bool tail = n_tail > 0 ? true : false;
  uint64_t counter_offset = 0;
  if (tail) {
    // If tail, we need generate randnum one more time, so here we add another `1`.
    counter_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1 + 1) * kVecSize;
    MaskAndScaleAddGpu<T, true><<<grid_size, kBlockSize, 0, ctx->cuda_stream()>>>(
        seed, cuda_gen_state, counter_offset, elem_cnt, rate, scale, n_tail, x, mask, addend, y,
        (x + tail_offset), (mask + tail_offset), (addend + tail_offset), (y + tail_offset));
  } else {
    counter_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize;
    MaskAndScaleAddGpu<T, false><<<grid_size, kBlockSize, 0, ctx->cuda_stream()>>>(
        seed, cuda_gen_state, counter_offset, elem_cnt, rate, scale, n_tail, x, mask, addend, y,
        (x + tail_offset), (mask + tail_offset), (addend + tail_offset), (y + tail_offset));
  }
}

template<typename T>
struct MaskAndScaleFunctor {
  OF_DEVICE_FUNC explicit MaskAndScaleFunctor(float scale) : scale(scale) {}
  OF_DEVICE_FUNC T operator()(T x, int8_t mask) const {
    return x * static_cast<T>(mask) * static_cast<T>(scale);
  }
  float scale;
};

template<typename T>
class DropoutKernelGPU final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  DropoutKernelGPU() = default;
  ~DropoutKernelGPU() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kGPU));
    return std::make_shared<FusedDropoutKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    auto* fused_dropout_kernel_state = dynamic_cast<FusedDropoutKernelState*>(state);
    CHECK_NOTNULL(fused_dropout_kernel_state);
    const auto& generator = fused_dropout_kernel_state->generator();
    CHECK_NOTNULL(generator);
    std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
        CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
    uint64_t seed = cuda_generator->current_seed();

    const float rate = ctx->Attr<float>("rate");
    float scale = 0.0;
    if (rate < 1.0f) { scale = 1.0f / (1.0f - rate); }
    one::CUDAGeneratorState* cuda_gen_state = cuda_generator->cuda_gen_state();

    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* addend = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      MaskAndScaleAdd<T>(ctx->device_ctx(), seed, cuda_gen_state, in->shape().elem_cnt(), rate,
                         scale, in->dptr<T>(), mask->mut_dptr<int8_t>(), addend->dptr<T>(),
                         out->mut_dptr<T>());
    } else {
      MaskAndScale<T>(ctx->device_ctx(), seed, cuda_gen_state, in->shape().elem_cnt(), rate, scale,
                      in->dptr<T>(), mask->mut_dptr<int8_t>(), out->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_KERNEL_GPU(dtype)                                                \
  REGISTER_USER_KERNEL("dropout").SetCreateFn<DropoutKernelGPU<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kGPU)                                      \
      && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)                    \
      && (user_op::HobDataType("mask", 0) == GetDataType<int8_t>::value));

REGISTER_DROPOUT_KERNEL_GPU(half)
REGISTER_DROPOUT_KERNEL_GPU(float)
REGISTER_DROPOUT_KERNEL_GPU(double)

template<typename T>
class DropoutGradKernelGPU final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  DropoutGradKernelGPU() = default;
  ~DropoutGradKernelGPU() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float scale = ctx->Attr<float>("scale");
    const int64_t elem_cnt = dy->shape().elem_cnt();
    OF_CUDA_CHECK((cuda::elementwise::Binary(MaskAndScaleFunctor<T>(scale), elem_cnt,
                                             dx->mut_dptr<T>(), dy->dptr<T>(), mask->dptr<int8_t>(),
                                             ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_GRAD_KERNEL_GPU(dtype)                                                 \
  REGISTER_USER_KERNEL("dropout_grad")                                                          \
      .SetCreateFn<DropoutGradKernelGPU<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                           \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_DROPOUT_GRAD_KERNEL_GPU(half)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(float)
REGISTER_DROPOUT_GRAD_KERNEL_GPU(double)

}  // namespace

}  // namespace oneflow
