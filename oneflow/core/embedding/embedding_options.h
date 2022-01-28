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
#ifndef ONEFLOW_EMBEDDING_EMBEDDING_OPTIONS_H_
#define ONEFLOW_EMBEDDING_EMBEDDING_OPTIONS_H_
#include "nlohmann/json.hpp"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {
namespace embedding {

enum class InitializerType {
  kUniform,
  kNormal,
};

struct EmbeddingInitializer {
  InitializerType type;
  union {
    struct {
      float low;
      float high;
    } uniform_param;
    struct {
      float mean;
      float std;
    } normal_param;
  };
};

struct EmbeddingColumn {
  EmbeddingInitializer initializer;
};

class EmbeddingOptions final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingOptions);
  EmbeddingOptions(std::string json_serialized) {
    auto json_object = nlohmann::json::parse(json_serialized);
    auto GetValue = [](const nlohmann::json& obj, const std::string& attr) -> nlohmann::json {
      nlohmann::json val = obj[attr];
      if (val == nlohmann::detail::value_t::null) { UNIMPLEMENTED(); }
      return val;
    };
    name_ = GetValue(json_object, "name");
    embedding_dim_ = GetValue(json_object, "embedding_dim");
    const int64_t scale_factor = GetValue(json_object, "scale_factor");
    line_size_ = embedding_dim_ * scale_factor;
    auto caches = json_object["cache"];
    if (caches != nlohmann::detail::value_t::null) {
      CHECK(caches.is_array());
      l1_cache_policy_ = GetValue(caches.at(0), "policy");
      l1_cache_memory_budget_mb_ = GetValue(caches.at(0), "cache_memory_budget_mb");
      l1_cache_value_memory_kind_ = GetValue(caches.at(0), "value_memory_kind");
      if (caches.size() > 1) {
        l2_cache_policy_ = GetValue(caches.at(1), "policy");
        l2_cache_memory_budget_mb_ = GetValue(caches.at(1), "cache_memory_budget_mb");
        l2_cache_value_memory_kind_ = GetValue(caches.at(1), "value_memory_kind");
      } else {
        l2_cache_policy_ = "none";
      }
    } else {
      l1_cache_policy_ = "none";
      l2_cache_policy_ = "none";
    }
    auto kv_store = GetValue(json_object, "kv_store");
    if (kv_store["persistent_table"] != nlohmann::detail::value_t::null) {
      auto persistent_table = kv_store["persistent_table"];
      persistent_table_path_ = GetValue(persistent_table, "path");
      persistent_table_phisical_block_size_ = GetValue(persistent_table, "physical_block_size");
    } else {
      UNIMPLEMENTED();
    }
    auto columns = json_object["columns"];
    if (columns != nlohmann::detail::value_t::null) {
      for (int32_t i = 0; i < columns.size(); ++i) {
        EmbeddingColumn embedding_column;
        auto column = columns.at(i);
        auto initializer = GetValue(column, "initializer");
        std::string type = GetValue(initializer, "type");
        if (type == "uniform") {
          embedding_column.initializer.type = InitializerType::kUniform;
          embedding_column.initializer.uniform_param.low = GetValue(initializer, "low");
          embedding_column.initializer.uniform_param.high = GetValue(initializer, "high");
        } else if (type == "normal") {
          embedding_column.initializer.type = InitializerType::kNormal;
          embedding_column.initializer.normal_param.mean = GetValue(initializer, "mean");
          embedding_column.initializer.normal_param.std = GetValue(initializer, "std");
        } else {
          UNIMPLEMENTED();
        }
        columns_.push_back(embedding_column);
      }
    } else {
      EmbeddingColumn embedding_column;
      auto initializer = GetValue(json_object, "default_initializer");
      std::string type = GetValue(initializer, "type");
      if (type == "uniform") {
        embedding_column.initializer.type = InitializerType::kUniform;
        embedding_column.initializer.uniform_param.low = GetValue(initializer, "low");
        embedding_column.initializer.uniform_param.high = GetValue(initializer, "high");
      } else if (type == "normal") {
        embedding_column.initializer.type = InitializerType::kNormal;
        embedding_column.initializer.normal_param.mean = GetValue(initializer, "mean");
        embedding_column.initializer.normal_param.std = GetValue(initializer, "std");
      } else {
        UNIMPLEMENTED();
      }
      columns_.push_back(embedding_column);
    }
  }
  ~EmbeddingOptions() = default;

  std::string Name() const { return name_; }
  int64_t EmbeddingSize() const { return embedding_dim_; }
  int64_t LineSize() const { return line_size_; }
  std::string L1CachePolicy() const { return l1_cache_policy_; }
  int64_t L1CacheMemoryBudgetMb() const { return l1_cache_memory_budget_mb_; }
  std::string L1CacheValueMemoryKind() const { return l1_cache_value_memory_kind_; }
  std::string L2CachePolicy() const { return l2_cache_policy_; }
  int64_t L2CacheMemoryBudgetMb() const { return l2_cache_memory_budget_mb_; }
  std::string L2CacheValueMemoryKind() const { return l2_cache_value_memory_kind_; }
  std::string PersistentTablePath() const { return persistent_table_path_; }
  int64_t PersistentTablePhysicalBlockSize() const { return persistent_table_phisical_block_size_; }
  std::vector<EmbeddingColumn> Columns() const { return columns_; }

 private:
  std::string name_;
  int64_t embedding_dim_;
  int64_t line_size_;
  std::string l1_cache_policy_;
  int64_t l1_cache_memory_budget_mb_;
  std::string l1_cache_value_memory_kind_;
  std::string l2_cache_policy_;
  int64_t l2_cache_memory_budget_mb_;
  std::string l2_cache_value_memory_kind_;
  std::string persistent_table_path_;
  int64_t persistent_table_phisical_block_size_;
  std::vector<EmbeddingColumn> columns_;
};

}  // namespace embedding
}  // namespace oneflow
#endif  // ONEFLOW_EMBEDDING_EMBEDDING_OPTIONS_H_
