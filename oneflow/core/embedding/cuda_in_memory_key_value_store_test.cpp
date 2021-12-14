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
#include "oneflow/core/embedding/cuda_in_memory_key_value_store.h"
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

void TestKeyValueStore(KeyValueStore* store, size_t num_embeddings, size_t test_embeddings,
                       size_t embedding_vec_size, size_t num_shards) {
  std::unique_ptr<ep::DeviceManagerRegistry> device_manager_registry(
      new ep::DeviceManagerRegistry());
  auto device = device_manager_registry->GetDevice(DeviceType::kCUDA, 0);
  ep::Stream* stream = device->CreateStream();

  uint64_t* keys = nullptr;
  float* values = nullptr;
  uint64_t* keys_host = nullptr;
  float* values_host = nullptr;
  uint64_t* context = nullptr;
  size_t keys_size = sizeof(uint64_t) * num_embeddings;
  size_t values_size = sizeof(float) * embedding_vec_size * num_embeddings;
  size_t context_size = sizeof(uint64_t) * num_embeddings;
  OF_CUDA_CHECK(cudaMalloc(&keys, keys_size));
  OF_CUDA_CHECK(cudaMalloc(&values, values_size));
  OF_CUDA_CHECK(cudaMalloc(&context, context_size));
  OF_CUDA_CHECK(cudaMallocHost(&keys_host, keys_size));
  OF_CUDA_CHECK(cudaMallocHost(&values_host, values_size));
  for (size_t i = 0; i < num_embeddings; ++i) {
    uint64_t key = i * num_shards + 3;
    keys_host[i] = key;
    for (size_t j = 0; j < embedding_vec_size; j++) {
      values_host[i * embedding_vec_size + j] = key;
    }
  }
  OF_CUDA_CHECK(cudaMemcpy(keys, keys_host, keys_size, cudaMemcpyDefault));
  OF_CUDA_CHECK(cudaMemcpy(values, values_host, values_size, cudaMemcpyDefault));
  const size_t batch_size = 128;
  for (size_t offset = 0; offset < test_embeddings; offset += batch_size) {
    const size_t num_keys = std::min(batch_size, test_embeddings - offset);
    store->Prefetch(stream, num_keys, keys + offset, context + offset);
    store->Update(stream, num_keys, keys + offset, context + offset,
                  values + offset * embedding_vec_size);
  }
  OF_CUDA_CHECK(cudaMemset(values_host, 0, values_size));
  OF_CUDA_CHECK(cudaMemset(values, 0, values_size));
  for (size_t offset = 0; offset < test_embeddings; offset += batch_size) {
    const size_t num_keys = std::min(batch_size, test_embeddings - offset);
    store->Prefetch(stream, num_keys, keys + offset, context + offset);
    store->Lookup(stream, num_keys, keys + offset, context + offset,
                  values + offset * embedding_vec_size);
  }
  OF_CUDA_CHECK(cudaMemcpy(values_host, values, values_size, cudaMemcpyDefault));
  OF_CUDA_CHECK(cudaDeviceSynchronize());
  for (size_t i = 0; i < test_embeddings; ++i) {
    uint64_t key = keys_host[i];
    for (size_t j = 0; j < embedding_vec_size; j++) {
      ASSERT_EQ(values_host[i * embedding_vec_size + j], key);
    }
  }
  OF_CUDA_CHECK(cudaDeviceSynchronize());
  OF_CUDA_CHECK(cudaGetLastError());
  OF_CUDA_CHECK(cudaFree(keys));
  OF_CUDA_CHECK(cudaFree(values));
  OF_CUDA_CHECK(cudaFreeHost(keys_host));
  OF_CUDA_CHECK(cudaFreeHost(values_host));

  CHECK_JUST(stream->Sync());
  device->DestroyStream(stream);
}

TEST(CudaInMemoryKeyValueStore, PlainEncoder) {
  if (!HasCudaDevice()) { return; }
  CudaInMemoryKeyValueStoreOptions options{};
  options.num_shards = 4;
  options.embedding_vec_size = 128;
  options.num_embeddings = 1024 * 4;
  options.num_device_embeddings = 1024;
  options.encoding_type = CudaInMemoryKeyValueStoreOptions::EncodingType::kPlain;
  std::unique_ptr<KeyValueStore> store = NewCudaInMemoryKeyValueStore(options);

  TestKeyValueStore(store.get(), options.num_embeddings, options.num_embeddings,
                    options.embedding_vec_size, options.num_shards);
}

#endif  // WITH_CUDA

}  // namespace

}  // namespace embedding

}  // namespace oneflow
