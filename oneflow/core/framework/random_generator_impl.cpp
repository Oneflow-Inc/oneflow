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

#include "oneflow/core/framework/random_generator_impl.h"

#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // WITH_CUDA

namespace oneflow {
namespace one {

#ifdef WITH_CUDA
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

CUDAGeneratorImpl::CUDAGeneratorImpl(uint64_t seed, int device_index)
    : DeviceGeneratorImpl(seed, detail::DeviceKey{DeviceType::kGPU, device_index}) {
  cudaDeviceProp prop;
  OF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  max_block_num_ = prop.multiProcessorCount;
  max_thread_num_ = GetThreadNum(prop);
  OF_CUDA_CHECK(
      cudaMalloc(&curand_states_, max_block_num_ * max_thread_num_ * sizeof(curandState)));
  detail::InitCurandStates(seed, max_block_num_, max_thread_num_, curand_states_);
}

CUDAGeneratorImpl::~CUDAGeneratorImpl() { OF_CUDA_CHECK(cudaFree(curand_states_)); }

void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  detail::InitCurandStates(seed_, max_block_num_, max_thread_num_, curand_states_);
}
#endif  // WITH_CUDA

namespace detail {

bool operator==(const DeviceKey& k1, const DeviceKey& k2) {
  return k1.device_type == k2.device_type && k1.device_index == k2.device_index;
}

size_t DeviceKeyHash::operator()(const DeviceKey& key) const {
  return (static_cast<uint64_t>(key.device_type) << 10) + key.device_index;
}

template<>
DeviceKey MakeDeviceKey<CPUGeneratorImpl>(int device_index) {
  return DeviceKey{DeviceType::kCPU, 0};
}

template<>
Maybe<CPUGeneratorImpl> MakeGeneratorImpl<CPUGeneratorImpl>(uint64_t seed, int device_index) {
  return std::make_shared<CPUGeneratorImpl>(seed);
}

#ifdef WITH_CUDA
int GetCudaDeviceCount() {
  /* static */ int cuda_device_count;
  OF_CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
  return cuda_device_count;
}

template<>
DeviceKey MakeDeviceKey<CUDAGeneratorImpl>(int device_index) {
  if (device_index == -1) { OF_CUDA_CHECK(cudaGetDevice(&device_index)); }
  return DeviceKey{DeviceType::kGPU, device_index};
}

template<>
Maybe<CUDAGeneratorImpl> MakeGeneratorImpl<CUDAGeneratorImpl>(uint64_t seed, int device_index) {
  CHECK_OR_RETURN(device_index >= 0 && device_index < GetCudaDeviceCount())
      << "Invalid device index " << device_index;
  return std::make_shared<CUDAGeneratorImpl>(seed, device_index);
}
#endif  // WITH_CUDA

}  // namespace detail

}  // namespace one
}  // namespace oneflow
