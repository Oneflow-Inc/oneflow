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
#include "oneflow/core/embedding/persistent_table_key_value_store.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/embedding/cached_key_value_store.h"

namespace oneflow {

namespace embedding {}  // namespace embedding

EmbeddingMgr::~EmbeddingMgr() {
  for (auto& pair : key_value_store_map_) { pair.second->SaveSnapshot("index"); }
}

embedding::KeyValueStore* EmbeddingMgr::GetKeyValueStore(
    const embedding::EmbeddingOptions& embedding_options, int64_t parallel_id,
    int64_t parallel_num) {
  const std::string& name = embedding_options.Name();
  const uint32_t line_size = embedding_options.LineSize();
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) { return it->second.get(); }

  std::unique_ptr<embedding::KeyValueStore> store;
  const std::string& path = embedding_options.PersistentTablePath();
  const std::string& num_rank = std::to_string(parallel_num);
  const int32_t rank_id_suffix_length = num_rank.size();
  const std::string& rank_id = std::to_string(parallel_id);
  embedding::PersistentTableKeyValueStoreOptions options{};
  options.table_options.path = path + "/" + std::string(rank_id_suffix_length - rank_id.size(), '0')
                               + rank_id + "-" + num_rank;
  options.table_options.value_size = line_size * GetSizeOfDataType(DataType::kFloat);
  options.table_options.key_size = GetSizeOfDataType(DataType::kInt64);
  options.table_options.physical_block_size = embedding_options.PersistentTablePhysicalBlockSize();
  options.table_options.target_chunk_size_mb = 4 * 1024;
  store = NewPersistentTableKeyValueStore(options);
  if (embedding_options.L2CachePolicy() != "none") {
    embedding::CacheOptions cache_options{};
    cache_options.value_memory_kind = embedding::CacheOptions::MemoryKind::kHost;
    if (embedding_options.L2CachePolicy() == "lru") {
      cache_options.policy = embedding::CacheOptions::Policy::kLRU;
    } else if (embedding_options.L2CachePolicy() == "full") {
      cache_options.policy = embedding::CacheOptions::Policy::kFull;
    } else {
      UNIMPLEMENTED();
    }
    cache_options.key_size = GetSizeOfDataType(DataType::kInt64);
    cache_options.value_size = GetSizeOfDataType(DataType::kFloat) * line_size;
    cache_options.capacity =
        embedding_options.L2CacheMemoryBudgetMb() * 1024 * 1024 / cache_options.value_size;
    std::unique_ptr<embedding::Cache> cache = embedding::NewCache(cache_options);
    LOG(ERROR) << "add L2 cache: " << embedding_options.L2CachePolicy() << " "
               << embedding_options.L2CacheMemoryBudgetMb();
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }
  if (embedding_options.L1CachePolicy() != "none") {
    embedding::CacheOptions cache_options{};
    cache_options.value_memory_kind = embedding::CacheOptions::MemoryKind::kDevice;
    if (embedding_options.L1CachePolicy() == "lru") {
      cache_options.policy = embedding::CacheOptions::Policy::kLRU;
    } else if (embedding_options.L1CachePolicy() == "full") {
      cache_options.policy = embedding::CacheOptions::Policy::kFull;
    } else {
      UNIMPLEMENTED();
    }
    cache_options.key_size = GetSizeOfDataType(DataType::kInt64);
    cache_options.value_size = GetSizeOfDataType(DataType::kFloat) * line_size;
    cache_options.capacity =
        embedding_options.L1CacheMemoryBudgetMb() * 1024 * 1024 / cache_options.value_size;
    std::unique_ptr<embedding::Cache> cache = embedding::NewCache(cache_options);
    LOG(ERROR) << "add L1 cache: " << embedding_options.L1CachePolicy() << " "
               << embedding_options.L1CacheMemoryBudgetMb();
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }

  store->ReserveQueryLength(embedding_options.MaxQueryLength());
  if (store->SnapshotExists("index")) { store->LoadSnapshot("index"); }
  auto pair = key_value_store_map_.emplace(map_key, std::move(store));
  CHECK(pair.second);
  return pair.first->second.get();
}

}  // namespace oneflow
