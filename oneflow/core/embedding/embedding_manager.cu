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

embedding::KeyValueStore* EmbeddingMgr::GetKeyValueStore(const std::string& name,
                                                         int64_t parallel_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  LOG(ERROR) << "GetKeyValueStore " << name << " " << parallel_id;
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) {
    LOG(ERROR) << "find " << name << " " << parallel_id;
    return it->second.get();
  }
  embedding::CudaInMemoryKeyValueStoreOptions options{};
  options.num_shards = 4;
  options.value_length = 128;
  options.num_keys = 1024 * 1024;
  options.num_device_keys = 1024 * 128;
  options.encoding_type = embedding::CudaInMemoryKeyValueStoreOptions::EncodingType::kOrdinal;
  std::unique_ptr<embedding::KeyValueStore> store = NewCudaInMemoryKeyValueStore(options);
  auto pair = key_value_store_map_.emplace(map_key, std::move(store));
  CHECK(pair.second);
  LOG(ERROR) << "not found, create " << name << " " << parallel_id;
  return pair.first->second.get();
}

}  // namespace oneflow
