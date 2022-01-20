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

struct EmbeddingInitializer {
  double mean;
  double scale;
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
    max_query_length_ = GetValue(json_object, "max_query_length");

    auto l1_cache = json_object["l1_cache"];
    if (l1_cache != nlohmann::detail::value_t::null) {
      l1_cache_policy_ = GetValue(l1_cache, "policy");  // python检查值范围
      l1_cache_memory_budget_mb_ = GetValue(l1_cache, "cache_memory_budget_mb");
    } else {
      l1_cache_policy_ = "none";
    }
    auto l2_cache = json_object["l2_cache"];
    if (l2_cache != nlohmann::detail::value_t::null) {
      l2_cache_policy_ = GetValue(l2_cache, "policy");  // python检查值范围
      l2_cache_memory_budget_mb_ = GetValue(l2_cache, "cache_memory_budget_mb");
    } else {
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
    auto optimizer = GetValue(json_object, "optimizer");
    optimizer_type_ = GetValue(optimizer, "type");
    if (optimizer_type_ == "sgd") {
      line_size_ = embedding_dim_;
    } else if (optimizer_type_ == "momentum") {
      beta_ = GetValue(optimizer, "beta");
      line_size_ = embedding_dim_ * 2;
    } else if (optimizer_type_ == "adam") {
      beta1_ = GetValue(optimizer, "beta1");
      beta2_ = GetValue(optimizer, "beta2");
      epsilon_ = GetValue(optimizer, "epsilon");
      do_bias_correction_ = GetValue(optimizer, "do_bias_correction");
      line_size_ = embedding_dim_ * 3;
    } else {
      UNIMPLEMENTED();
    }
    auto learning_rate_schedule = GetValue(json_object, "learning_rate_schedule");
    learning_rate_ = GetValue(learning_rate_schedule, "learning_rate");

    auto warmup = learning_rate_schedule["warmup"];
    if (warmup != nlohmann::detail::value_t::null) {
      warmup_type_ = GetValue(warmup, "type");
      if (warmup_type_ == "linear") {
        warmup_conf_.mutable_linear_conf()->set_warmup_batches(GetValue(warmup, "warmup_batches"));
        warmup_conf_.mutable_linear_conf()->set_start_multiplier(
            GetValue(warmup, "start_multiplier"));
      } else if (warmup_type_ == "constant") {
        warmup_conf_.mutable_constant_conf()->set_warmup_batches(
            GetValue(warmup, "warmup_batches"));
        warmup_conf_.mutable_constant_conf()->set_multiplier(GetValue(warmup, "multiplier"));
      } else {
        UNIMPLEMENTED();
      }
    } else {
      warmup_type_ = "none";
    }

    auto learning_rate_decay = learning_rate_schedule["learning_rate_decay"];
    if (learning_rate_decay != nlohmann::detail::value_t::null) {
      learning_rate_decay_type_ = GetValue(learning_rate_decay, "type");
      if (learning_rate_decay_type_ == "polynomial") {
        learning_rate_decay_conf_.mutable_polynomial_conf()->set_decay_batches(
            GetValue(learning_rate_decay, "decay_batches"));
        learning_rate_decay_conf_.mutable_polynomial_conf()->set_end_learning_rate(
            GetValue(learning_rate_decay, "end_learning_rate"));
        learning_rate_decay_conf_.mutable_polynomial_conf()->set_power(
            GetValue(learning_rate_decay, "power"));
        learning_rate_decay_conf_.mutable_polynomial_conf()->set_cycle(
            GetValue(learning_rate_decay, "cycle"));
      } else {
        UNIMPLEMENTED();
      }
    } else {
      learning_rate_decay_type_ = "none";
    }

    auto columns = json_object["columns"];
    if (columns != nlohmann::detail::value_t::null) {
      for (int32_t i = 0; i < columns.size(); ++i) {
        EmbeddingColumn embedding_column;
        auto column = columns.at(i);
        auto initializer = GetValue(column, "initializer");
        embedding_column.initializer.mean = GetValue(initializer, "mean");
        embedding_column.initializer.scale = GetValue(initializer, "scale");
        columns_.push_back(embedding_column);
      }
    } else {
      EmbeddingColumn embedding_column;
      embedding_column.initializer.mean = 0;
      embedding_column.initializer.scale = 1;
      columns_.push_back(embedding_column);
    }
  }
  ~EmbeddingOptions() = default;

  std::string Name() const { return name_; }
  int64_t EmbeddingSize() const { return embedding_dim_; }
  int64_t LineSize() const { return line_size_; }
  int64_t MaxQueryLength() const { return max_query_length_; }
  std::string L1CachePolicy() const { return l1_cache_policy_; }
  int64_t L1CacheMemoryBudgetMb() const { return l1_cache_memory_budget_mb_; }
  std::string L2CachePolicy() const { return l2_cache_policy_; }
  int64_t L2CacheMemoryBudgetMb() const { return l2_cache_memory_budget_mb_; }
  std::string PersistentTablePath() const { return persistent_table_path_; }
  int64_t PersistentTablePhysicalBlockSize() const { return persistent_table_phisical_block_size_; }
  std::string Optimizer() const { return optimizer_type_; }
  float Beta() const { return beta_; }
  float Beta1() const { return beta1_; }
  float Beta2() const { return beta2_; }
  float Epsilon() const { return epsilon_; }
  bool DoBiasCorrection() const { return do_bias_correction_; }

  float LearningRate() const { return learning_rate_; }
  std::string WarmupType() const { return warmup_type_; }
  WarmupConf WarmupConfProto() const { return warmup_conf_; }
  std::string LearningRateDecayType() const { return learning_rate_decay_type_; }
  LearningRateDecayConf LearningRateDecayConfProto() const { return learning_rate_decay_conf_; }
  std::vector<EmbeddingColumn> Columns() const { return columns_; }

 private:
  std::string name_;
  int64_t embedding_dim_;
  int64_t line_size_;
  int64_t max_query_length_;
  std::string l1_cache_policy_;
  int64_t l1_cache_memory_budget_mb_;
  std::string l2_cache_policy_;
  int64_t l2_cache_memory_budget_mb_;
  std::string persistent_table_path_;
  int64_t persistent_table_phisical_block_size_;
  std::string optimizer_type_;
  float learning_rate_;
  float beta_;
  float beta1_;
  float beta2_;
  float epsilon_;
  bool do_bias_correction_;
  std::string warmup_type_;
  WarmupConf warmup_conf_;
  std::string learning_rate_decay_type_;
  LearningRateDecayConf learning_rate_decay_conf_;
  std::vector<EmbeddingColumn> columns_;
};

}  // namespace embedding
}  // namespace oneflow
#endif  // ONEFLOW_EMBEDDING_EMBEDDING_OPTIONS_H_
