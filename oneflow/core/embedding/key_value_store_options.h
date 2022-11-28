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
#ifndef ONEFLOW_EMBEDDING_KEY_VALUE_STORE_OPTIONS_H_
#define ONEFLOW_EMBEDDING_KEY_VALUE_STORE_OPTIONS_H_
#include "nlohmann/json.hpp"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/embedding/cache.h"

namespace oneflow {
namespace embedding {

namespace {

void ParseCacheOptions(const nlohmann::json& cache_obj, CacheOptions* cache_options) {
  CHECK_GT(cache_options->key_size, 0);
  CHECK_GT(cache_options->value_size, 0);
  CHECK(cache_obj.contains("policy"));
  CHECK(cache_obj["policy"].is_string());
  std::string policy = cache_obj["policy"].get<std::string>();
  if (policy == "lru") {
    cache_options->policy = CacheOptions::Policy::kLRU;
  } else if (policy == "full") {
    cache_options->policy = CacheOptions::Policy::kFull;
  } else {
    UNIMPLEMENTED() << "Unsupported cache policy";
  }
  int64_t capacity = 0;
  if (cache_obj.contains("capacity")) {
    CHECK(cache_obj["capacity"].is_number());
    capacity = cache_obj["capacity"].get<int64_t>();
  }
  if (cache_obj.contains("cache_memory_budget_mb")) {
    CHECK(cache_obj["cache_memory_budget_mb"].is_number());
    int64_t cache_memory_budget_mb = cache_obj["cache_memory_budget_mb"].get<int64_t>();
    if (cache_memory_budget_mb > 0) {
      CHECK_EQ(capacity, 0) << "when set capacity, must not set cache_memory_budget_mb";
      capacity = cache_memory_budget_mb * 1024 * 1024 / cache_options->value_size;
    }
  }
  CHECK_GT(capacity, 0) << "capacity or cache_memory_budget_mb must be set";
  // add an extra_capacity to avoid crash by uneven partition.
  const int64_t extra_capacity = capacity * 0.05;
  cache_options->capacity = capacity + (extra_capacity > 4096 ? extra_capacity : 4096);
  CHECK(cache_obj.contains("value_memory_kind"));
  CHECK(cache_obj["value_memory_kind"].is_string());
  std::string value_memory_kind = cache_obj["value_memory_kind"].get<std::string>();
  if (value_memory_kind == "device") {
    cache_options->value_memory_kind = CacheOptions::MemoryKind::kDevice;
  } else if (value_memory_kind == "host") {
    cache_options->value_memory_kind = CacheOptions::MemoryKind::kHost;
  } else {
    UNIMPLEMENTED() << "Unsupported cache value_memory_kind";
  }
}

}  // namespace

class KeyValueStoreOptions final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyValueStoreOptions);
  explicit KeyValueStoreOptions(const std::string& json_serialized) {
    auto json_object = nlohmann::json::parse(json_serialized);

    CHECK(json_object.contains("key_type_size"));
    CHECK(json_object["key_type_size"].is_number());
    key_type_size_ = json_object["key_type_size"].get<int64_t>();

    CHECK(json_object.contains("value_type_size"));
    CHECK(json_object["value_type_size"].is_number());
    std::string value_type_name = json_object["value_type"];
    if (value_type_name == "oneflow.float" || value_type_name == "oneflow.float32") {
      value_type_ = DataType::kFloat;
    } else {
      UNIMPLEMENTED();
    }
    value_type_size_ = json_object["value_type_size"].get<int64_t>();

    CHECK(json_object.contains("parallel_num"));
    CHECK(json_object["parallel_num"].is_number());
    const int64_t parallel_num = json_object["parallel_num"].get<int64_t>();

    CHECK(json_object.contains("name"));
    CHECK(json_object["name"].is_string());
    name_ = json_object["name"].get<std::string>();

    CHECK(json_object.contains("storage_dim"));
    CHECK(json_object["storage_dim"].is_number());
    line_size_ = json_object["storage_dim"].get<int64_t>();

    CHECK(json_object.contains("kv_store"));
    auto kv_store = json_object["kv_store"];

    auto caches = kv_store["caches"];
    if (caches != nlohmann::detail::value_t::null && caches.size() > 0) {
      CHECK(caches.is_array());
      cache_options_.resize(caches.size());
      for (int i = 0; i < caches.size(); ++i) {
        cache_options_.at(i).key_size = key_type_size_;
        cache_options_.at(i).value_size = value_type_size_ * line_size_;
        cache_options_.at(i).value_type = value_type_;
        ParseCacheOptions(caches.at(i), &cache_options_.at(i));
      }
    }

    CHECK(kv_store.contains("persistent_table"));
    auto persistent_table = kv_store["persistent_table"];
    CHECK(persistent_table.contains("path"));
    auto path = persistent_table["path"];
    CHECK(path.is_array() || path.is_string());
    if (path.is_array()) {
      CHECK_EQ(path.size(), parallel_num);
      for (int i = 0; i < path.size(); ++i) {
        CHECK(path.at(i).is_string());
        persistent_table_paths_.push_back(path.at(i).get<std::string>());
      }
    } else {
      std::string root_path = path.get<std::string>();
      const std::string& num_rank = std::to_string(parallel_num);
      const int64_t rank_id_suffix_length = num_rank.size();
      for (int i = 0; i < parallel_num; ++i) {
        const std::string& rank_id = std::to_string(i);
        const std::string rank_i_path = root_path + "/"
                                        + std::string(rank_id_suffix_length - rank_id.size(), '0')
                                        + rank_id + "-" + num_rank;
        persistent_table_paths_.push_back(rank_i_path);
      }
    }
    CHECK(persistent_table.contains("physical_block_size"));
    CHECK(persistent_table["physical_block_size"].is_number());
    persistent_table_physical_block_size_ = persistent_table["physical_block_size"].get<int64_t>();
    if (persistent_table.contains("capacity_hint")) {
      CHECK(persistent_table["capacity_hint"].is_number());
      persistent_table_capacity_hint_ = persistent_table["capacity_hint"].get<int64_t>();
    } else {
      persistent_table_capacity_hint_ = 0;
    }
  }
  ~KeyValueStoreOptions() = default;
  int64_t KeyTypeSize() const { return key_type_size_; }
  int64_t ValueTypeSize() const { return value_type_size_; }
  DataType ValueType() const { return value_type_; }
  const std::string& Name() const { return name_; }
  int64_t LineSize() const { return line_size_; }
  const std::vector<CacheOptions>& GetCachesOptions() const { return cache_options_; }
  const std::vector<std::string>& PersistentTablePaths() const { return persistent_table_paths_; }
  int64_t PersistentTablePhysicalBlockSize() const { return persistent_table_physical_block_size_; }
  int64_t PersistentTableCapacityHint() const { return persistent_table_capacity_hint_; }
  bool IsFullCache() const {
    if (cache_options_.size() > 0 && cache_options_.at(0).policy == CacheOptions::Policy::kFull) {
      return true;
    }
    return false;
  }

 private:
  int64_t key_type_size_;
  int64_t value_type_size_;
  DataType value_type_;
  std::string name_;
  int64_t line_size_;
  std::vector<std::string> persistent_table_paths_;
  int64_t persistent_table_physical_block_size_;
  int64_t persistent_table_capacity_hint_;
  std::vector<CacheOptions> cache_options_;
};

}  // namespace embedding
}  // namespace oneflow
#endif  // ONEFLOW_EMBEDDING_KEY_VALUE_STORE_OPTIONS_H_
