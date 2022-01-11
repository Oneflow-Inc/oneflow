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
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/embedding/block_based_key_value_store.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace embedding {}  // namespace embedding

EmbeddingMgr::~EmbeddingMgr() {
  const uint32_t line_size = ParseIntegerFromEnv("EMBEDDING_SIZE", 128);
  for (const auto& key7device_id : device_id_map_) {
    CudaCurrentDeviceGuard guard(key7device_id.second);
    auto device = Global<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA,
                                                                      key7device_id.second);
    CHECK(device);
    auto* stream = device->CreateStream();
    auto* cuda_stream = stream->As<ep::CudaStream>();
    auto* cache = cache_map_.at(key7device_id.first).get();
    auto* store = key_value_store_map_.at(key7device_id.first).get();
    uint32_t* n_dumped = nullptr;
    uint32_t* host_n_dumped = nullptr;
    void* dumped_keys = nullptr;
    void* dumped_values = nullptr;
    const uint32_t max_n_dump_keys = 65536 * 26;
    OF_CUDA_CHECK(cudaMalloc(&n_dumped, sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMallocHost(&host_n_dumped, sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMalloc(&dumped_keys, max_n_dump_keys * GetSizeOfDataType(DataType::kInt64)));
    OF_CUDA_CHECK(cudaMalloc(&dumped_values,
                             max_n_dump_keys * line_size * GetSizeOfDataType(DataType::kFloat)));
    uint64_t cache_capacity = cache->Capacity();
    for (uint64_t start_key_index = 0; start_key_index < cache_capacity;
         start_key_index += max_n_dump_keys) {
      cache->Dump(stream, start_key_index,
                  std::min(start_key_index + max_n_dump_keys, cache_capacity), n_dumped,
                  dumped_keys, dumped_values);
      OF_CUDA_CHECK(cudaMemcpyAsync(host_n_dumped, n_dumped, sizeof(uint32_t), cudaMemcpyDefault,
                                    cuda_stream->cuda_stream()));
      CHECK_JUST(stream->Sync());
      if (*host_n_dumped == 0) { continue; }
      store->Put(stream, *host_n_dumped, dumped_keys, dumped_values, nullptr);
      CHECK_JUST(stream->Sync());
    }
    CHECK_JUST(stream->Sync());
    OF_CUDA_CHECK(cudaFree(n_dumped));
    OF_CUDA_CHECK(cudaFreeHost(host_n_dumped));
    OF_CUDA_CHECK(cudaFree(dumped_keys));
    OF_CUDA_CHECK(cudaFree(dumped_values));
    device->DestroyStream(stream);
  }
}

embedding::Cache* EmbeddingMgr::GetCache(const embedding::EmbeddingOptions& embedding_options,
                                         int64_t parallel_id, int64_t parallel_num) {
  const std::string& name = embedding_options.EmbeddingName();
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  int device_id = 0;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  std::unique_lock<std::mutex> lock(mutex_);
  auto device_id_it = device_id_map_.find(map_key);
  if (device_id_it == device_id_map_.end()) {
    device_id_map_[map_key] = device_id;
  } else {
    CHECK_EQ(device_id_it->second, device_id);
  }
  auto it = cache_map_.find(map_key);
  if (it != cache_map_.end()) {
    CHECK_EQ(device_id_map_.at(map_key), device_id);
    return it->second.get();
  }
  embedding::CudaLruCacheOptions options{};
  const uint32_t line_size = embedding_options.EmbeddingSize();
  options.line_size = line_size;
  options.memory_budget_mb = embedding_options.CacheMemoryBudgetMb();
  CHECK_GT(options.memory_budget_mb, 0);
  options.max_query_length = 65536 * 26;
  options.key_type = DataType::kInt64;
  options.value_type = DataType::kFloat;
  std::unique_ptr<embedding::Cache> cache = embedding::NewCudaLruCache(options);
  auto pair = cache_map_.emplace(map_key, std::move(cache));
  CHECK(pair.second);
  return pair.first->second.get();
}

embedding::KeyValueStore* EmbeddingMgr::GetKeyValueStore(
    const embedding::EmbeddingOptions& embedding_options, int64_t parallel_id,
    int64_t parallel_num) {
  const std::string& name = embedding_options.EmbeddingName();
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  int device_id = 0;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  std::unique_lock<std::mutex> lock(mutex_);
  auto device_id_it = device_id_map_.find(map_key);
  if (device_id_it == device_id_map_.end()) {
    device_id_map_[map_key] = device_id;
  } else {
    CHECK_EQ(device_id_it->second, device_id);
  }
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) { return it->second.get(); }

  std::unique_ptr<embedding::KeyValueStore> store;
  const std::string& kv_store = embedding_options.KVStore();
  if (kv_store == "cuda_in_memory") {
    embedding::CudaInMemoryKeyValueStoreOptions options{};
    options.num_shards = 4;
    options.value_length = embedding_options.EmbeddingSize();
    options.num_keys = embedding_options.NumKeys();
    CHECK_GT(options.num_keys, 0);
    options.num_device_keys = embedding_options.NumDeviceKeys();
    options.key_type = DataType::kInt64;
    options.value_type = DataType::kFloat;
    options.encoding_type = embedding::CudaInMemoryKeyValueStoreOptions::EncodingType::kOrdinal;
    store = NewCudaInMemoryKeyValueStore(options);
  } else if (kv_store == "block_based") {
    const std::string& path = embedding_options.FixedTablePath();
    const std::string& num_rank = std::to_string(parallel_num);
    const int32_t rank_id_suffix_length = num_rank.size();
    const std::string& rank_id = std::to_string(parallel_id);
    embedding::BlockBasedKeyValueStoreOptions options{};
    options.path = path + "/" + std::string(rank_id_suffix_length - rank_id.size(), '0') + rank_id
                   + "_" + num_rank;
    options.value_length = embedding_options.EmbeddingSize();
    options.key_type = DataType::kInt64;
    options.value_type = DataType::kFloat;
    options.max_query_length = 65536 * 26;
    options.block_size = embedding_options.FixedTableBlockSize();
    store = NewBlockBasedKeyValueStore(options);
  } else {
    UNIMPLEMENTED();
  }
  auto pair = key_value_store_map_.emplace(map_key, std::move(store));
  CHECK(pair.second);
  return pair.first->second.get();
}

}  // namespace oneflow
