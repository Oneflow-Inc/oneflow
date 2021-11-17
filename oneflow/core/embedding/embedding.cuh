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
#ifndef ONEFLOW_CORE_EMBEDDING_EMBEDDING_H_
#define ONEFLOW_CORE_EMBEDDING_EMBEDDING_H_

#include <cub/cub.cuh>
#include <cassert>
#include <curand_kernel.h>
#include "oneflow/core/embedding/cuda_cache.cuh"
#include "oneflow/core/embedding/mapped_parameter_server.h"
#include "oneflow/core/embedding/hash_function.cuh"

namespace oneflow {

namespace embedding {

namespace {

template<typename Key, typename Elem, typename Idx>
__global__ void PostParameterServerLookup(uint64_t n_elem_per_value, const Key* keys, Elem* values,
                                          const Idx* ps_indices, const Idx* n_ps_found,
                                          const Idx* ps_found_indices, const Elem* ps_found_values,
                                          const Idx* n_ps_miss, const Idx* ps_miss_indices) {
  const uint64_t n_found_elem = *n_ps_found * n_elem_per_value;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_found_elem;
       i += gridDim.x * blockDim.x) {
    const uint64_t row = i / n_elem_per_value;
    const uint64_t col = i - row * n_elem_per_value;
    values[ps_indices[ps_found_indices[row]] * n_elem_per_value + col] = ps_found_values[i];
  }
  const uint64_t n_miss_elem = *n_ps_miss * n_elem_per_value;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_miss_elem;
       i += gridDim.x * blockDim.x) {
    const uint64_t row = i / n_elem_per_value;
    const uint64_t col = i - row * n_elem_per_value;
    const uint64_t key_idx = ps_indices[ps_miss_indices[row]];
    curandStatePhilox4_32_10_t state;
    curand_init(0, keys[key_idx], col, &state);
    values[ps_indices[ps_miss_indices[row]] * n_elem_per_value + col] = curand_uniform(&state);
  }
}

template<typename Key, typename Elem, typename Idx>
__global__ void PostParameterServerPrefetch(uint64_t n_elem_per_value, const Key* keys,
                                            Elem* values, const Idx* n_ps_found,
                                            const Idx* ps_found_indices,
                                            const Elem* ps_found_values, const Idx* n_ps_miss,
                                            const Idx* ps_miss_indices) {
  const uint64_t n_found_elem = *n_ps_found * n_elem_per_value;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_found_elem;
       i += gridDim.x * blockDim.x) {
    const uint64_t row = i / n_elem_per_value;
    const uint64_t col = i - row * n_elem_per_value;
    values[ps_found_indices[row] * n_elem_per_value + col] = ps_found_values[i];
  }
  const uint64_t n_miss_elem = *n_ps_miss * n_elem_per_value;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_miss_elem;
       i += gridDim.x * blockDim.x) {
    const uint64_t row = i / n_elem_per_value;
    const uint64_t col = i - row * n_elem_per_value;
    const uint64_t key_idx = ps_miss_indices[row];
    curandStatePhilox4_32_10_t state;
    curand_init(0, keys[key_idx], col, &state);
    values[ps_miss_indices[row] * n_elem_per_value + col] = curand_uniform(&state);
  }
}

template<typename T, typename U>
__global__ void CopyKernel(T* dst, const T* src, U* n) {
  U size_n = *n;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_n;
       i += gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

}  // namespace

template<typename Key, typename Elem, typename Hash, typename Idx>
void LookupParameterServerFunc(void* params);

template<typename Key, typename Elem, typename Hash, typename Idx>
void UpdateParameterServerFunc(void* params);

template<typename Key, typename Elem, typename Hash, typename Idx>
class Embedding {
 public:
  Embedding(uint32_t log2_capacity, uint64_t n_elem_per_value, const std::string& base_dir,
            uint32_t log2_n_set, uint64_t max_n_key)
      : n_elem_per_value_(n_elem_per_value), max_n_key_(max_n_key) {
    parameter_server_.reset(
        new MappedParameterServer<Key, Elem, Idx>(log2_capacity, n_elem_per_value, base_dir));
    uint64_t cache_workspace_size =
        CacheContext<Key, Elem, Idx>::GetWorkspaceSize(log2_n_set, n_elem_per_value);
    cudaMalloc(&cache_workspace_, cache_workspace_size);
    cache_.reset(new CacheContext<Key, Elem, Idx>(log2_n_set, n_elem_per_value, cache_workspace_,
                                                  cache_workspace_size));
    InitKernel<Key, Elem, Idx><<<1024, 1024>>>(*cache_);
    cudaMallocHost(&n_host_, sizeof(Idx));
    cudaMalloc(&n_device_, sizeof(Idx));
    cudaMalloc(&values_devices_, sizeof(Elem) * max_n_key * n_elem_per_value);
    cudaMallocHost(&keys_host_, sizeof(Key) * max_n_key);
    cudaMalloc(&keys_device_, sizeof(Key) * max_n_key);
    cudaMallocHost(&values_host_, sizeof(Elem) * max_n_key * n_elem_per_value);
    cudaMalloc(&values_device_, sizeof(Elem) * max_n_key * n_elem_per_value);
    cudaMallocHost(&indices_host_, sizeof(Idx) * max_n_key);
    cudaMallocHost(&indices_device_, sizeof(Idx) * max_n_key);
    cudaMallocHost(&n_pull_found_host_, sizeof(Idx));
    cudaMallocHost(&pull_found_indices_host_, sizeof(Idx) * max_n_key);
    cudaMallocHost(&n_pull_miss_host_, sizeof(Idx));
    cudaMallocHost(&pull_miss_indices_host_, sizeof(Idx) * max_n_key);
  }

  cudaError_t Prefetch(const Idx* n_key, const Key* keys, cudaStream_t stream) {
    cudaMemsetAsync(n_device_, 0, sizeof(Idx), stream);
    LookupKernel<Key, Elem, Idx, XXH64, true>
        <<<(max_n_key_ + 128 - 1) / 128, dim3(32, 4), 0, stream>>>(*cache_, n_key, keys, nullptr,
                                                                   n_device_, keys_host_, nullptr);
    cudaMemcpyAsync(n_host_, n_device_, sizeof(Idx), cudaMemcpyDefault, stream);
    cudaError_t err =
        cudaLaunchHostFunc(stream, LookupParameterServerFunc<Key, Elem, Hash, Idx>, this);
    if (err != cudaSuccess) { return err; }
    PostParameterServerPrefetch<Key, Elem, Idx><<<(max_n_key_ + 128 - 1) / 128, 128, 0, stream>>>(
        n_elem_per_value_, keys_host_, values_device_, n_pull_found_host_, pull_found_indices_host_,
        values_host_, n_pull_miss_host_, pull_miss_indices_host_);
    CopyKernel<<<(max_n_key_ + 128 - 1) / 128, 128, 0, stream>>>(keys_device_, keys_host_,
                                                                 n_device_);
    cudaMemsetAsync(n_device_, 0, sizeof(Idx), stream);
    UpdateKernel<Key, Elem, Idx, XXH64, 128>
        <<<(max_n_key_ + 128 - 1) / 128, dim3(32, 4), 0, stream>>>(
            *cache_, n_host_, keys_device_, values_device_, n_device_, keys_host_, values_host_);
    cudaMemcpyAsync(n_host_, n_device_, sizeof(Idx), cudaMemcpyDefault, stream);
    cudaLaunchHostFunc(stream, UpdateParameterServerFunc<Key, Elem, Hash, Idx>, this);
    return cudaSuccess;
  }

  cudaError_t Lookup(const Idx* n_key, const Key* keys, Elem* values, cudaStream_t stream) {
    cudaMemsetAsync(n_device_, 0, sizeof(Idx), stream);
    LookupKernel<Key, Elem, Idx, XXH64, false>
        <<<(max_n_key_ + 128 - 1) / 128, dim3(32, 4), 0, stream>>>(
            *cache_, n_key, keys, values, n_device_, keys_host_, indices_device_);
    cudaMemcpyAsync(n_host_, n_device_, sizeof(Idx), cudaMemcpyDefault, stream);
    cudaError_t err =
        cudaLaunchHostFunc(stream, LookupParameterServerFunc<Key, Elem, Hash, Idx>, this);
    if (err != cudaSuccess) { return err; }
    PostParameterServerLookup<Key, Elem, Idx><<<(max_n_key_ + 128 - 1) / 128, 128, 0, stream>>>(
        n_elem_per_value_, keys, values, indices_device_, n_pull_found_host_,
        pull_found_indices_host_, values_host_, n_pull_miss_host_, pull_miss_indices_host_);
    return cudaSuccess;
  }

  cudaError_t Update(const Idx* n_key, const Key* keys, const Elem* values, cudaStream_t stream) {
    cudaMemsetAsync(n_device_, 0, sizeof(Idx), stream);
    UpdateKernel<Key, Elem, Idx, XXH64, 128>
        <<<(max_n_key_ + 128 - 1) / 128, dim3(32, 4), 0, stream>>>(
            *cache_, n_key, keys, values, n_device_, keys_host_, values_host_);
    cudaMemcpyAsync(n_host_, n_device_, sizeof(Idx), cudaMemcpyDefault, stream);
    return cudaLaunchHostFunc(stream, UpdateParameterServerFunc<Key, Elem, Hash, Idx>, this);
  }

  void LookupParameterServer() {
    *n_pull_found_host_ = 0;
    *n_pull_miss_host_ = 0;
    parameter_server_->Pull(n_host_, keys_host_, n_pull_found_host_, pull_found_indices_host_,
                            values_host_, n_pull_miss_host_, pull_miss_indices_host_);
  }

  void UpdateParameterServer() { parameter_server_->Push(n_host_, keys_host_, values_host_); }

 private:
  std::unique_ptr<CacheContext<Key, Elem, Idx>> cache_;
  std::unique_ptr<MappedParameterServer<Key, Elem, Idx>> parameter_server_;
  Idx* n_host_;
  Idx* n_device_;
  Elem* values_devices_;
  Key* keys_host_;
  Key* keys_device_;
  Idx* indices_host_;
  Idx* indices_device_;
  Elem* values_host_;
  Elem* values_device_;
  Idx* n_pull_found_host_;
  Idx* pull_found_indices_host_;
  Idx* n_pull_miss_host_;
  Idx* pull_miss_indices_host_;
  uint64_t max_n_key_;
  uint64_t n_elem_per_value_;
  void* cache_workspace_;
};

template<typename Key, typename Elem, typename Hash, typename Idx>
void LookupParameterServerFunc(void* params) {
  auto* embedding = reinterpret_cast<Embedding<Key, Elem, Hash, Idx>*>(params);
  embedding->LookupParameterServer();
}

template<typename Key, typename Elem, typename Hash, typename Idx>
void UpdateParameterServerFunc(void* params) {
  auto* embedding = reinterpret_cast<Embedding<Key, Elem, Hash, Idx>*>(params);
  embedding->UpdateParameterServer();
}

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_EMBEDDING_H_
