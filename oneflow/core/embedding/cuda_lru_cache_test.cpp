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
#include "oneflow/core/embedding/cuda_lru_cache.h"
#include "oneflow/core/device/cuda_util.h"
#include <gtest/gtest.h>
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace embedding {

namespace {

#ifdef WITH_CUDA

bool HasCudaDevice() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) { return false; }
  if (device_count <= 0) { return false; }
  return true;
}

TEST(CudaLruCache, CudaLruCache) {
  if (!HasCudaDevice()) { return; }

  std::unique_ptr<ep::DeviceManagerRegistry> device_manager_registry(
      new ep::DeviceManagerRegistry());
  auto device = device_manager_registry->GetDevice(DeviceType::kCUDA, 0);
  ep::Stream* stream = device->CreateStream();

  CudaLruCacheOptions options{};
  const uint32_t line_size = 128;
  options.line_size = line_size;
  options.log2_n_set = 8;
  options.max_query_length = 65536;
  options.key_type = DataType::kInt64;
  options.value_type = DataType::kFloat;
  std::unique_ptr<Cache> cache(NewCudaLruCache(options));

  std::unordered_set<int64_t> in_cache;
  const size_t n_iter = 128;
  const uint32_t n_keys = 1024;
  int64_t* d_keys;
  int64_t* keys;
  uint32_t* d_n_missing;
  uint32_t* n_missing;
  int64_t* d_missing_keys;
  int64_t* missing_keys;
  uint32_t* d_missing_indices;
  uint32_t* missing_indices;
  float* d_values;
  float* values;
  float* d_evicted_values;
  float* evicted_values;
  uint32_t* d_n_evicted;
  uint32_t* n_evicted;
  int64_t* d_evicted_keys;
  int64_t* evicted_keys;
  const size_t keys_size = n_keys * sizeof(int64_t);
  OF_CUDA_CHECK(cudaMalloc(&d_keys, keys_size));
  OF_CUDA_CHECK(cudaMallocHost(&keys, keys_size));
  OF_CUDA_CHECK(cudaMalloc(&d_n_missing, sizeof(uint32_t)));
  OF_CUDA_CHECK(cudaMallocHost(&n_missing, sizeof(uint32_t)));
  OF_CUDA_CHECK(cudaMalloc(&d_missing_keys, keys_size));
  OF_CUDA_CHECK(cudaMallocHost(&missing_keys, keys_size));
  const size_t indices_size = n_keys * sizeof(uint32_t);
  OF_CUDA_CHECK(cudaMalloc(&d_missing_indices, indices_size));
  OF_CUDA_CHECK(cudaMallocHost(&missing_indices, indices_size));
  const size_t values_size = n_keys * line_size * sizeof(float);
  OF_CUDA_CHECK(cudaMalloc(&d_values, values_size));
  OF_CUDA_CHECK(cudaMallocHost(&values, values_size));
  OF_CUDA_CHECK(cudaMalloc(&d_evicted_values, values_size));
  OF_CUDA_CHECK(cudaMallocHost(&evicted_values, values_size));
  OF_CUDA_CHECK(cudaMalloc(&d_n_evicted, sizeof(uint32_t)));
  OF_CUDA_CHECK(cudaMallocHost(&n_evicted, sizeof(uint32_t)));
  OF_CUDA_CHECK(cudaMalloc(&d_evicted_keys, keys_size));
  OF_CUDA_CHECK(cudaMallocHost(&evicted_keys, keys_size));
  std::vector<int64_t> random_keys(n_keys * 32);
  std::iota(random_keys.begin(), random_keys.end(), 1);
  std::random_device rd;
  std::mt19937 g(rd());
  for (size_t iter = 0; iter < n_iter; ++iter) {
    std::shuffle(random_keys.begin(), random_keys.end(), g);
    std::copy(random_keys.begin(), random_keys.begin() + n_keys, keys);
    uint32_t expect_n_missing = 0;
    std::unordered_set<int64_t> expect_missing_keys_set;
    std::unordered_set<uint32_t> expect_missing_indices_set;
    std::unordered_set<int64_t> keys_set;
    for (size_t i = 0; i < n_keys; ++i) {
      keys_set.emplace(keys[i]);
      if (in_cache.count(keys[i]) == 0) {
        expect_missing_keys_set.emplace(keys[i]);
        expect_missing_indices_set.emplace(i);
        expect_n_missing += 1;
      }
    }
    // test
    OF_CUDA_CHECK(cudaMemcpy(d_keys, keys, keys_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    cache->Test(stream, n_keys, d_keys, d_n_missing, d_missing_keys, d_missing_indices);
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    OF_CUDA_CHECK(cudaMemcpy(n_missing, d_n_missing, sizeof(uint32_t), cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaMemcpy(missing_keys, d_missing_keys, keys_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaMemcpy(missing_indices, d_missing_indices, indices_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(*n_missing, expect_n_missing);
    std::unordered_set<int64_t> test_missing_keys_set;
    std::unordered_set<uint32_t> test_missing_indices_set;
    for (size_t i = 0; i < *n_missing; ++i) {
      test_missing_keys_set.emplace(missing_keys[i]);
      test_missing_indices_set.emplace(missing_indices[i]);
      ASSERT_EQ(keys[missing_indices[i]], missing_keys[i]);
    }
    ASSERT_EQ(test_missing_keys_set, expect_missing_keys_set);
    ASSERT_EQ(test_missing_indices_set, expect_missing_indices_set);

    // get
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    cache->Get(stream, n_keys, d_keys, d_values, d_n_missing, d_missing_keys, d_missing_indices);
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    OF_CUDA_CHECK(cudaMemcpy(n_missing, d_n_missing, sizeof(uint32_t), cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaMemcpy(missing_keys, d_missing_keys, keys_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaMemcpy(missing_indices, d_missing_indices, indices_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaMemcpy(values, d_values, values_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(*n_missing, expect_n_missing);
    std::unordered_set<int64_t> get_missing_keys_set;
    std::unordered_set<uint32_t> get_missing_indices_set;
    for (size_t i = 0; i < *n_missing; ++i) {
      get_missing_keys_set.emplace(missing_keys[i]);
      get_missing_indices_set.emplace(missing_indices[i]);
      ASSERT_EQ(keys[missing_indices[i]], missing_keys[i]);
    }
    ASSERT_EQ(get_missing_keys_set, expect_missing_keys_set);
    ASSERT_EQ(get_missing_indices_set, expect_missing_indices_set);
    for (size_t i = 0; i < n_keys; ++i) {
      if (get_missing_keys_set.count(keys[i]) == 0) {
        for (size_t j = 0; j < line_size; ++j) {
          ASSERT_EQ(values[i * line_size + j], static_cast<float>(keys[i]))
              << "iter " << iter << " i " << i << " j " << j;
        }
      }
    }

    // put
    for (size_t i = 0; i < n_keys; ++i) {
      for (size_t j = 0; j < line_size; ++j) {
        values[i * line_size + j] = static_cast<float>(keys[i]);
      }
    }
    OF_CUDA_CHECK(cudaMemcpy(d_values, values, values_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    cache->Put(stream, n_keys, d_keys, d_values, d_n_evicted, d_evicted_keys, d_evicted_values);
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    OF_CUDA_CHECK(cudaMemcpy(n_evicted, d_n_evicted, sizeof(uint32_t), cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaMemcpy(evicted_keys, d_evicted_keys, keys_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaMemcpy(evicted_values, d_evicted_values, values_size, cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t i = 0; i < *n_evicted; ++i) {
      ASSERT_TRUE(in_cache.count(evicted_keys[i]) > 0 || keys_set.count(evicted_keys[i]) > 0);
      for (size_t j = 0; j < line_size; ++j) {
        ASSERT_EQ(evicted_values[i * line_size + j], static_cast<float>(evicted_keys[i]));
      }
    }
    for (size_t i = 0; i < n_keys; ++i) { in_cache.emplace(keys[i]); }
    for (size_t i = 0; i < *n_evicted; ++i) { in_cache.erase(evicted_keys[i]); }
  }
  OF_CUDA_CHECK(cudaFree(d_keys));
  OF_CUDA_CHECK(cudaFreeHost(keys));
  OF_CUDA_CHECK(cudaFree(d_n_missing));
  OF_CUDA_CHECK(cudaFreeHost(n_missing));
  OF_CUDA_CHECK(cudaFree(d_missing_keys));
  OF_CUDA_CHECK(cudaFreeHost(missing_keys));
  OF_CUDA_CHECK(cudaFree(d_missing_indices));
  OF_CUDA_CHECK(cudaFreeHost(missing_indices));
  OF_CUDA_CHECK(cudaFree(d_values));
  OF_CUDA_CHECK(cudaFreeHost(values));
  OF_CUDA_CHECK(cudaFree(d_evicted_values));
  OF_CUDA_CHECK(cudaFreeHost(evicted_values));
  OF_CUDA_CHECK(cudaFree(d_n_evicted));
  OF_CUDA_CHECK(cudaFreeHost(n_evicted));
  OF_CUDA_CHECK(cudaFree(d_evicted_keys));
  OF_CUDA_CHECK(cudaFreeHost(evicted_keys));
  device->DestroyStream(stream);
}

#endif  // WITH_CUDA

}  // namespace

}  // namespace embedding

}  // namespace oneflow
