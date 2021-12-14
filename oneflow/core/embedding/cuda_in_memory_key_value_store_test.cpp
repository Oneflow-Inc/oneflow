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

TEST(CudaInMemoryKeyValueStore, PlainEncoder) {
  int device_count = 0;
  if (cudaGetDevice(&device_count) != cudaSuccess) { return; }
  if (device_count <= 0) { return; }

  std::unique_ptr<ep::DeviceManagerRegistry> device_manager_registry(
      new ep::DeviceManagerRegistry());
  auto device = device_manager_registry->GetDevice(DeviceType::kCUDA, 0);
  ep::Stream* stream = device->CreateStream();

  CudaInMemoryKeyValueStoreOptions options{};
  options.num_shards = 4;
  options.embedding_vec_size = 128;
  options.num_embeddings = 1024 * 4;
  options.num_device_embeddings = 1024;
  options.encoding_type = CudaInMemoryKeyValueStoreOptions::EncodingType::kPlain;
  std::unique_ptr<KeyValueStore> store = NewCudaInMemoryKeyValueStore(options);

  uint64_t* keys = nullptr;
  float* values = nullptr;
  uint64_t* keys_host = nullptr;
  float* values_host = nullptr;
  uint64_t* context = nullptr;
  size_t keys_size = sizeof(uint64_t) * options.num_embeddings;
  size_t values_size = sizeof(float) * options.embedding_vec_size * options.num_embeddings;
  size_t context_size = sizeof(uint64_t) * options.num_embeddings;
  OF_CUDA_CHECK(cudaMalloc(&keys, keys_size));
  OF_CUDA_CHECK(cudaMalloc(&values, values_size));
  OF_CUDA_CHECK(cudaMalloc(&context, context_size));
  OF_CUDA_CHECK(cudaMallocHost(&keys_host, keys_size));
  OF_CUDA_CHECK(cudaMallocHost(&values_host, values_size));
  for (size_t i = 0; i < options.num_embeddings; ++i) {
    uint64_t key = i * options.num_shards + 3;
    keys_host[i] = key;
    for (size_t j = 0; j < options.embedding_vec_size; j++) {
      values_host[i * options.embedding_vec_size + j] = key;
    }
  }
  OF_CUDA_CHECK(cudaMemcpy(keys, keys_host, keys_size, cudaMemcpyDefault));
  OF_CUDA_CHECK(cudaMemcpy(values, values_host, values_size, cudaMemcpyDefault));
  size_t batch_size = 128;
  for (size_t offset = 0; offset < options.num_embeddings; offset += batch_size) {
    store->Prefetch(stream, batch_size, keys + offset, context + offset);
    store->Update(stream, batch_size, keys + offset, context + offset, values + offset);
  }
  for (size_t offset = 0; offset < options.num_embeddings; offset += batch_size) {
    store->Prefetch(stream, batch_size, keys + offset, context + offset);
    store->Lookup(stream, batch_size, keys + offset, context + offset, values + offset);
  }
  OF_CUDA_CHECK(cudaDeviceSynchronize());
  OF_CUDA_CHECK(cudaGetLastError());
  OF_CUDA_CHECK(cudaFree(keys));
  OF_CUDA_CHECK(cudaFree(values));
  OF_CUDA_CHECK(cudaFree(keys_host));
  OF_CUDA_CHECK(cudaFree(values_host));

  CHECK_JUST(stream->Sync());
  device->DestroyStream(stream);
}

#endif  // WITH_CUDA

}  // namespace

}  // namespace embedding

}  // namespace oneflow
