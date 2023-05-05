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
#ifdef WITH_CUDA

#include "oneflow/core/ep/cuda/cuda_random_generator.h"

#include "oneflow/core/common/util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/device/cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace oneflow {
namespace ep {

namespace {

int GetThreadNum(const cudaDeviceProp& prop) {
  switch (prop.major) {
    case 3:  // Kepler
      return 2 * 192;
    case 5:  // Maxwell
      return 2 * 128;
    case 6:  // Pascal
      if ((prop.minor == 1) || (prop.minor == 2)) {
        return 2 * 128;
      } else {
        return 2 * 64;
      }
    case 7:  // Volta and Turing
      return 2 * 64;
    default: return 2 * 64;
  }
}

}  // namespace

CUDAGenerator::CUDAGenerator(uint64_t seed, int device_index)
    : RandomGenerator(), seed_(seed), device_index_(device_index), philox_offset_per_thread_(0) {
  int device_count;
  OF_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  CHECK_LT_OR_THROW(device_index, device_count)
      << "only " << device_count << " cuda devices are visible.";
  cudaDeviceProp prop;  // NOLINT
  OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
  max_block_num_ = prop.multiProcessorCount;
  max_thread_num_ = GetThreadNum(prop);
}

void CUDAGenerator::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

std::tuple<uint64_t, dim3, dim3> CUDAGenerator::CalcExecutionPolicy(int64_t total_elements,
                                                                    ep::CudaStream* stream) {
  // NOTE(Liang Depeng): the implementation is modified from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/DistributionTemplates.h

  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = 256;  // block_size_bound
  // number of randoms given by distributions like curand_uniform4, curand_uniform2_double
  // used in calculating philox offset.
  const uint32_t curand4_engine_calls = 4;
  const uint32_t unroll = curand4_engine_calls;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  uint32_t blocks_per_sm = stream->device_properties().maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(stream->device_properties().multiProcessorCount) * blocks_per_sm,
      grid.x);
  // number of times random will be generated per thread, to offset philox counter in thc random
  // state
  uint64_t counter_offset =
      ((numel - 1) / (block_size * grid.x * unroll) + 1) * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);
}

// NOTE(Liang Depeng): The implementation of ` CUDAGenerator::get_philox_offset` is modified
// from
//      https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/CUDAGenerator.cpp#L269
//      in order to make distribution related cuda kernels to have the same output as pytorch
//      when setting the same seed.
uint64_t CUDAGenerator::get_philox_offset(uint64_t increment) {
  std::lock_guard<std::mutex> lock(mutex_);
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  CHECK_EQ(this->philox_offset_per_thread_ % 4, 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return offset;
}

// NOTE: The RNG state comprises the seed, and an offset used for Philox.
// The following line is just here for aligning Pytorch and it is also no
// practical effect in Pytorch just for backward compatibility reason.
// For more details pls refer to:
// https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/cuda/CUDAGenerator.cpp#L152
static constexpr size_t states_size = 200 * sizeof(4120);
static constexpr size_t seed_size = sizeof(uint64_t);
static constexpr size_t offset_size = sizeof(int64_t);
static constexpr size_t total_size = states_size + seed_size + offset_size;

size_t CUDAGenerator::GetStateSize() const { return total_size; }

void CUDAGenerator::GetState(size_t state_size, void* state) const {
  CHECK_EQ_OR_THROW(state_size, GetStateSize())
      << "the state size of cuda generator should be equal to " << GetStateSize();
  memset(static_cast<uint8_t*>(state), -1, states_size);
  memcpy(static_cast<uint8_t*>(state) + states_size, &seed_, seed_size);
  memcpy(static_cast<uint8_t*>(state) + states_size + seed_size, &philox_offset_per_thread_,
         offset_size);
}

void CUDAGenerator::SetState(size_t state_size, const void* state) {
  CHECK_EQ_OR_THROW(state_size, GetStateSize())
      << "the state size of cuda generator should be equal to " << GetStateSize();
  const uint8_t* data = static_cast<const uint8_t*>(state);
  seed_ = *((uint64_t*)(data + states_size));
  philox_offset_per_thread_ = *((uint64_t*)(data + states_size + seed_size));
}

template<>
std::string GetRandomGeneratorDeviceTypeName<CUDAGenerator>() {
  return "cuda";
}

}  // namespace ep
}  // namespace oneflow

#endif  // WITH_CUDA
