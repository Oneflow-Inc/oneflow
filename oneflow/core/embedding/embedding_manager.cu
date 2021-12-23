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

namespace oneflow {

namespace embedding {}  // namespace embedding

embedding::Cache* EmbeddingMgr::GetCache(const std::string& name, int64_t parallel_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = cache_map_.find(map_key);
  if (it != cache_map_.end()) { return it->second.get(); }
  embedding::CudaLruCacheOptions options{};
  const uint32_t line_size = 128;
  options.line_size = line_size;
  options.log2_n_set = 19;
  options.max_query_length = 65536 * 26;
  options.key_type = DataType::kInt64;
  options.value_type = DataType::kFloat;
  std::unique_ptr<embedding::Cache> cache = embedding::NewCudaLruCache(options);
  auto pair = cache_map_.emplace(map_key, std::move(cache));
  CHECK(pair.second);
  return pair.first->second.get();
}

embedding::KeyValueStore* EmbeddingMgr::GetKeyValueStore(const std::string& name,
                                                         int64_t parallel_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) { return it->second.get(); }
  embedding::CudaInMemoryKeyValueStoreOptions options{};
  options.num_shards = 4;
  options.value_length = 128;
  options.num_keys = 1024 * 1024 * 128 / 4;
  options.num_device_keys = 0;
  options.key_type = DataType::kInt64;
  options.value_type = DataType::kFloat;
  options.encoding_type = embedding::CudaInMemoryKeyValueStoreOptions::EncodingType::kOrdinal;
  std::unique_ptr<embedding::KeyValueStore> store = NewCudaInMemoryKeyValueStore(options);
  auto pair = key_value_store_map_.emplace(map_key, std::move(store));
  CHECK(pair.second);
  return pair.first->second.get();
}

}  // namespace oneflow
