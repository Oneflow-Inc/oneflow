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
#include "oneflow/core/embedding/persistent_table_key_value_store.h"
#include "oneflow/core/embedding/cached_key_value_store.h"
#include "oneflow/core/embedding/mock_key_value_store.h"
#include "oneflow/core/embedding/cache.h"
#include "oneflow/core/device/cuda_util.h"
#include <gtest/gtest.h>
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/embedding/posix_file.h"

namespace oneflow {

namespace embedding {

namespace {

#ifdef WITH_CUDA

std::string CreateTempDirectory() {
  const char* tmp_env = getenv("TMPDIR");
  const char* tmp_dir = tmp_env == nullptr ? "/tmp" : tmp_env;
  std::string tpl = std::string(tmp_dir) + "/test_kv_XXXXXX";
  char* path = mkdtemp(const_cast<char*>(tpl.c_str()));
  PCHECK(path != nullptr);
  return std::string(path);
}

bool HasCudaDevice() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) { return false; }
  if (device_count <= 0) { return false; }
  return true;
}

void TestKeyValueStore(KeyValueStore* store, size_t num_embeddings, size_t test_embeddings,
                       size_t embedding_vec_size) {
  auto device = Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA, 0);
  ep::Stream* stream = device->CreateStream();

  store->SaveSnapshot("init");

  uint64_t* keys = nullptr;
  float* values = nullptr;
  float* values1 = nullptr;
  uint64_t* keys_host = nullptr;
  float* values_host = nullptr;
  uint64_t* context = nullptr;
  uint32_t* n_missing = nullptr;
  uint32_t* host_n_missing = nullptr;
  uint64_t* missing_keys = nullptr;
  uint32_t* missing_indices = nullptr;
  size_t keys_size = sizeof(uint64_t) * num_embeddings;
  size_t values_size = sizeof(float) * embedding_vec_size * num_embeddings;
  size_t context_size = sizeof(uint64_t) * num_embeddings;
  const size_t batch_size = 128;
  OF_CUDA_CHECK(cudaMalloc(&keys, keys_size));
  OF_CUDA_CHECK(cudaMalloc(&values, values_size));
  OF_CUDA_CHECK(cudaMalloc(&values1, values_size));
  OF_CUDA_CHECK(cudaMalloc(&context, context_size));
  OF_CUDA_CHECK(cudaMallocHost(&keys_host, keys_size));
  OF_CUDA_CHECK(cudaMallocHost(&values_host, values_size));
  OF_CUDA_CHECK(cudaMallocHost(&host_n_missing, sizeof(uint32_t)));
  OF_CUDA_CHECK(cudaMalloc(&missing_keys, batch_size * sizeof(uint64_t)));
  OF_CUDA_CHECK(cudaMalloc(&missing_indices, batch_size * sizeof(uint32_t)));
  OF_CUDA_CHECK(cudaMalloc(&n_missing, sizeof(uint32_t)));
  for (size_t i = 0; i < num_embeddings; ++i) {
    uint64_t key = i + 1;
    keys_host[i] = key;
    for (size_t j = 0; j < embedding_vec_size; j++) {
      values_host[i * embedding_vec_size + j] = key;
    }
  }
  OF_CUDA_CHECK(cudaMemcpy(keys, keys_host, keys_size, cudaMemcpyDefault));
  OF_CUDA_CHECK(cudaMemcpy(values, values_host, values_size, cudaMemcpyDefault));

  store->Put(stream, 0, keys, values);
  OF_CUDA_CHECK(cudaDeviceSynchronize());
  OF_CUDA_CHECK(cudaGetLastError());

  for (size_t offset = 0; offset < test_embeddings; offset += batch_size) {
    const size_t num_keys = std::min(batch_size, test_embeddings - offset);
    store->Get(stream, num_keys, keys + offset, values1 + offset * embedding_vec_size, n_missing,
               missing_indices);
    OF_CUDA_CHECK(cudaMemcpy(host_n_missing, n_missing, sizeof(uint32_t), cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(*host_n_missing, num_keys);
    store->Put(stream, num_keys, keys + offset, values + offset * embedding_vec_size);
  }

  OF_CUDA_CHECK(cudaDeviceSynchronize());

  store->SaveSnapshot("final");

  OF_CUDA_CHECK(cudaMemset(values_host, 0, values_size));
  OF_CUDA_CHECK(cudaMemset(values, 0, values_size));
  for (size_t offset = 0; offset < test_embeddings; offset += batch_size) {
    const size_t num_keys = std::min(batch_size, test_embeddings - offset);
    store->Get(stream, num_keys, keys + offset, values + offset * embedding_vec_size, n_missing,
               missing_indices);
    OF_CUDA_CHECK(cudaMemcpy(host_n_missing, n_missing, sizeof(uint32_t), cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(*host_n_missing, 0);
  }
  OF_CUDA_CHECK(cudaMemcpy(values_host, values, values_size, cudaMemcpyDefault));
  OF_CUDA_CHECK(cudaDeviceSynchronize());
  for (size_t i = 0; i < test_embeddings; ++i) {
    uint64_t key = keys_host[i];
    for (size_t j = 0; j < embedding_vec_size; j++) {
      ASSERT_EQ(values_host[i * embedding_vec_size + j], key);
    }
  }

  store->LoadSnapshot("init");

  for (size_t offset = 0; offset < test_embeddings; offset += batch_size) {
    const size_t num_keys = std::min(batch_size, test_embeddings - offset);
    store->Get(stream, num_keys, keys + offset, values1 + offset * embedding_vec_size, n_missing,
               missing_indices);
    OF_CUDA_CHECK(cudaMemcpy(host_n_missing, n_missing, sizeof(uint32_t), cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(*host_n_missing, num_keys);
  }

  store->LoadSnapshot("final");

  OF_CUDA_CHECK(cudaMemset(values_host, 0, values_size));
  OF_CUDA_CHECK(cudaMemset(values, 0, values_size));
  for (size_t offset = 0; offset < test_embeddings; offset += batch_size) {
    const size_t num_keys = std::min(batch_size, test_embeddings - offset);
    store->Get(stream, num_keys, keys + offset, values + offset * embedding_vec_size, n_missing,
               missing_indices);
    OF_CUDA_CHECK(cudaMemcpy(host_n_missing, n_missing, sizeof(uint32_t), cudaMemcpyDefault));
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(*host_n_missing, 0);
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
  OF_CUDA_CHECK(cudaFree(values1));
  OF_CUDA_CHECK(cudaFreeHost(keys_host));
  OF_CUDA_CHECK(cudaFreeHost(values_host));
  OF_CUDA_CHECK(cudaFreeHost(host_n_missing));
  OF_CUDA_CHECK(cudaFree(n_missing));
  OF_CUDA_CHECK(cudaFree(missing_keys));
  OF_CUDA_CHECK(cudaFree(missing_indices));
  CHECK_JUST(stream->Sync());
  device->DestroyStream(stream);
}

TEST(PersistentTableKeyValueStore, PersistentTableKeyValueStore) {
  if (!HasCudaDevice()) { return; }
  Singleton<ep::DeviceManagerRegistry>::New();
  PersistentTableKeyValueStoreOptions options{};
  uint32_t value_length = 128;

  std::string path = CreateTempDirectory();
  options.table_options.path = path;
  options.table_options.value_size = value_length * sizeof(float);
  options.table_options.key_size = GetSizeOfDataType(DataType::kUInt64);
  options.table_options.physical_block_size = 512;

  std::unique_ptr<KeyValueStore> store = NewPersistentTableKeyValueStore(options);
  store->ReserveQueryLength(128);
  TestKeyValueStore(store.get(), 1024, 1024, value_length);
  store.reset();
  PosixFile::RecursiveDelete(path);
  Singleton<ep::DeviceManagerRegistry>::Delete();
}

TEST(CachedKeyValueStore, LRU) {
  if (!HasCudaDevice()) { return; }
  Singleton<ep::DeviceManagerRegistry>::New();
  PersistentTableKeyValueStoreOptions store_options{};
  std::string path = CreateTempDirectory();
  store_options.table_options.path = path;
  uint32_t value_length = 128;
  store_options.table_options.value_size = value_length * sizeof(float);
  store_options.table_options.key_size = GetSizeOfDataType(DataType::kUInt64);
  store_options.table_options.physical_block_size = 512;
  std::unique_ptr<KeyValueStore> store = NewPersistentTableKeyValueStore(store_options);
  CacheOptions cache_options{};
  cache_options.policy = CacheOptions::Policy::kLRU;
  cache_options.value_memory_kind = CacheOptions::MemoryKind::kDevice;
  cache_options.value_size = 512;
  cache_options.capacity = 512;
  cache_options.key_size = 8;
  std::unique_ptr<Cache> cache = NewCache(cache_options);
  std::unique_ptr<KeyValueStore> cached_store =
      NewCachedKeyValueStore(std::move(store), std::move(cache));
  cached_store->ReserveQueryLength(128);
  TestKeyValueStore(cached_store.get(), 1024, 1024, value_length);
  cached_store.reset();
  PosixFile::RecursiveDelete(path);
  Singleton<ep::DeviceManagerRegistry>::Delete();
}

TEST(CachedKeyValueStore, Full) {
  if (!HasCudaDevice()) { return; }
  Singleton<ep::DeviceManagerRegistry>::New();
  PersistentTableKeyValueStoreOptions store_options{};
  std::string path = CreateTempDirectory();
  store_options.table_options.path = path;
  uint32_t value_length = 128;
  store_options.table_options.value_size = value_length * sizeof(float);
  store_options.table_options.key_size = GetSizeOfDataType(DataType::kUInt64);
  store_options.table_options.physical_block_size = 512;
  std::unique_ptr<KeyValueStore> store = NewPersistentTableKeyValueStore(store_options);
  CacheOptions cache_options{};
  cache_options.policy = CacheOptions::Policy::kFull;
  cache_options.value_memory_kind = CacheOptions::MemoryKind::kHost;
  cache_options.value_size = 512;
  cache_options.capacity = 1024 * 2;
  cache_options.key_size = 8;
  std::unique_ptr<Cache> cache = NewCache(cache_options);
  std::unique_ptr<KeyValueStore> cached_store =
      NewCachedKeyValueStore(std::move(store), std::move(cache));
  cached_store->ReserveQueryLength(128);
  TestKeyValueStore(cached_store.get(), 1024, 1024, value_length);
  cached_store.reset();
  PosixFile::RecursiveDelete(path);
  Singleton<ep::DeviceManagerRegistry>::Delete();
}

TEST(MockKeyValueStore, Mock) {
  if (!HasCudaDevice()) { return; }
  Singleton<ep::DeviceManagerRegistry>::New();
  MockKeyValueStoreOptions store_options{};
  std::string path = CreateTempDirectory();
  uint32_t value_length = 128;
  store_options.value_size = value_length * sizeof(float);
  store_options.key_size = GetSizeOfDataType(DataType::kUInt64);
  std::unique_ptr<KeyValueStore> store = NewMockKeyValueStore(store_options);
  store->ReserveQueryLength(128);
  TestKeyValueStore(store.get(), 1024, 1024, value_length);
  store.reset();
  PosixFile::RecursiveDelete(path);
  Singleton<ep::DeviceManagerRegistry>::Delete();
}

#endif  // WITH_CUDA

}  // namespace

}  // namespace embedding

}  // namespace oneflow
