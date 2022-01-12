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
#include "oneflow/core/embedding/fixed_table_key_value_store.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/embedding/cached_key_value_store.h"

namespace oneflow {

namespace embedding {}  // namespace embedding

EmbeddingMgr::~EmbeddingMgr() = default;

embedding::KeyValueStore* EmbeddingMgr::GetKeyValueStore(
    const embedding::EmbeddingOptions& embedding_options, int64_t parallel_id,
    int64_t parallel_num) {
  const std::string& name = embedding_options.EmbeddingName();
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) { return it->second.get(); }

  embedding::CudaLruCacheOptions cache_options{};
  uint32_t line_size = embedding_options.LineSize();
  cache_options.memory_budget_mb = embedding_options.CacheMemoryBudgetMb();
  CHECK_GT(cache_options.memory_budget_mb, 0);
  cache_options.max_query_length = 65536 * 26;
  cache_options.key_size = GetSizeOfDataType(DataType::kInt64);
  cache_options.value_size = GetSizeOfDataType(DataType::kFloat) * line_size;
  std::unique_ptr<embedding::Cache> cache = embedding::NewCudaLruCache(cache_options);

  std::unique_ptr<embedding::KeyValueStore> store;
  const std::string& kv_store = embedding_options.KVStore();
  if (kv_store == "cuda_in_memory") {
    embedding::CudaInMemoryKeyValueStoreOptions options{};
    options.num_shards = 4;
    options.value_length = line_size;
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
    embedding::FixedTableKeyValueStoreOptions options{};
    options.table_options.path = path + "/"
                                 + std::string(rank_id_suffix_length - rank_id.size(), '0')
                                 + rank_id + "_" + num_rank;
    options.table_options.value_size = line_size * GetSizeOfDataType(DataType::kFloat);
    options.table_options.key_size = GetSizeOfDataType(DataType::kInt64);
    options.max_query_length = 65536 * 26;
    options.table_options.block_size = embedding_options.FixedTableBlockSize();
    options.table_options.num_blocks_per_chunk = 4 * 1024 * 1024;
    store = NewFixedTableKeyValueStore(options);
  } else {
    UNIMPLEMENTED();
  }
  std::unique_ptr<embedding::KeyValueStore> cached_store =
      NewCachedKeyValueStore(std::move(store), std::move(cache));
  auto pair = key_value_store_map_.emplace(map_key, std::move(cached_store));
  CHECK(pair.second);
  return pair.first->second.get();
}

}  // namespace oneflow
