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
#include "oneflow/user/kernels/random_mask_generator.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/cuda/atomic.cuh"

namespace oneflow {

namespace {

constexpr int32_t kMinPackPerThread = 2;

using PackType = ulonglong2;

union Pack {
  PackType p_value;
  int8_t b_value[sizeof(PackType)];
};

__device__ int8_t GenMask(curandState* state, const float rate) {
  return curand_uniform(state) > rate;
}

// __global__ void GenerateGpu(curandState* state, const int64_t n, const float rate, int8_t* mask) {
//   const int id = blockIdx.x * blockDim.x + threadIdx.x;
//   curandState localState = state[id];
//   PackType* pack_mask = reinterpret_cast<PackType*>(mask);
//   Pack pack;
//   CUDA_1D_KERNEL_LOOP(i, n / sizeof(PackType)) {
// #pragma unroll
//     for (int j = 0; j < sizeof(PackType); ++j) { pack.b_value[j] = GenMask(&localState, rate); }
//     pack_mask[i] = pack.p_value;
//   }
//   const int32_t rem_cnt = n % sizeof(PackType);
//   const int32_t rem_offset = n - rem_cnt;
//   if (id < rem_cnt) { mask[id + rem_offset] = GenMask(&localState, rate); }
//   state[id] = localState;
// }

constexpr int32_t kVecSize = 4;
constexpr int32_t kBlockSize = 256;

union RandPack4 {
  float4 storage;
  float elem[4];
};

template<bool tail>
__global__ void GenerateGpu(uint64_t seed, one::CUDAGeneratorState* cuda_gen_state, 
                            uint64_t inc_offset, const int64_t elem_cnt, 
                            const float rate, int64_t n_tail, int8_t* mask, int8_t* tail_mask) {
  using MaskType = cuda::elementwise::PackType<int8_t, 4>;
  using MaskPack = cuda::elementwise::Pack<int8_t, 4>;
  
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);

  RandPack4 rand_uniform_pack4;

  for (int64_t linear_index = global_thread_id * 4; linear_index < elem_cnt;
    linear_index += gridDim.x * blockDim.x * 4) {
    rand_uniform_pack4.storage = curand_uniform4(&state);

    MaskPack mask_vec;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      mask_vec.elem[i] = rand_uniform_pack4.elem[i] > rate;
    }

    *(reinterpret_cast<MaskType*>(mask + linear_index)) = mask_vec.storage;
    }

    if (tail && global_thread_id < n_tail) {
      const float rand_uniform = curand_uniform(&state);
      const int8_t mask_val = rand_uniform > rate;
      tail_mask[global_thread_id] = mask_val;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      int32_t new_counter = cuda::atomic::Add(&cuda_gen_state->dev_counter, 1) + 1;
      if (new_counter == gridDim.x) {
        cuda_gen_state->dev_counter = 0;           // reset counter to zero
        cuda_gen_state->dev_offset += inc_offset;  // maintain the state of generator's dev_offset
      }
    }
}

}  // namespace

unsigned int ComputeGridSize(const int32_t block_size, const int64_t elem_cnt) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
  unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
  grid_size = std::min((unsigned int)prop.multiProcessorCount * blocks_per_sm, grid_size);
  return grid_size;
}

void RandomMaskGenerator<DeviceType::kCUDA>::Generate(ep::Stream* stream, const int64_t elem_cnt,
                                                      const float rate, int8_t* mask) {
  // int32_t block_num = generator_->max_block_num();
  // int32_t thread_num = generator_->max_thread_num();
  // auto* curand_states = generator_->curand_states();
  // const int32_t elem_cnt_per_block = thread_num * sizeof(PackType) * kMinPackPerThread;
  // const int32_t block_num_final =
  //     std::min(static_cast<int32_t>((n + elem_cnt_per_block - 1) / elem_cnt_per_block), block_num);
  // GenerateGpu<<<block_num_final, thread_num, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
  //     curand_states, n, rate, mask);
  printf("Here>>??? \n"); 
  unsigned int grid_size = ComputeGridSize(kBlockSize, elem_cnt);
  constexpr int pack_size = 4;
  const int64_t pack_num = elem_cnt / pack_size;
  const int64_t tail_offset = pack_num * pack_size;
  const int64_t n_tail = elem_cnt - tail_offset;
  const bool tail = n_tail > 0 ? true : false;
  uint64_t inc_offset = 0;
  
  uint64_t seed = generator_->current_seed();
  one::CUDAGeneratorState* cuda_gen_state = generator_->cuda_gen_state();
  
  if (tail) {
    // If tail, we need generate randnum one more time, so here we add another `1`.
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize + 1;
    GenerateGpu<true>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            seed, cuda_gen_state, inc_offset, elem_cnt, rate, n_tail, mask, mask+tail_offset);
  } else {
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize;
    GenerateGpu<false>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            seed, cuda_gen_state, inc_offset, elem_cnt, rate, n_tail, mask, mask+tail_offset);
  }

}

template class RandomMaskGenerator<DeviceType::kCUDA>;

}  // namespace oneflow
