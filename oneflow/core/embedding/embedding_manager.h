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
#ifndef ONEFLOW_CORE_EMBEDDING_EMBEDDING_MANAGER_H_
#define ONEFLOW_CORE_EMBEDDING_EMBEDDING_MANAGER_H_

#include "oneflow/core/device/cuda_util.h"

#include "oneflow/core/embedding/cuda_lru_cache.h"
#include "oneflow/core/embedding/cuda_in_memory_key_value_store.h"

namespace oneflow {

namespace embedding {}  // namespace embedding

class EmbeddingMgr final {
 public:
  EmbeddingMgr() = default;
  ~EmbeddingMgr() = default;

  embedding::Cache* GetCache(const std::string& name, int64_t parallel_id);

  embedding::KeyValueStore* GetKeyValueStore(const std::string& name, int64_t parallel_id);

 private:
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<embedding::Cache>> cache_map_;
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<embedding::KeyValueStore>>
      key_value_store_map_;
  std::mutex mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_EMBEDDING_MANAGER_H_
