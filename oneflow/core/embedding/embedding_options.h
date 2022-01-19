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

class EmbeddingOptions final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingOptions);
  EmbeddingOptions(std::string json_serialized) {
    auto json_object = nlohmann::json::parse(json_serialized);
    embedding_name_ = json_object["embedding_name"];
    embedding_size_ = json_object["embedding_size"];
    l1_cache_policy_ = json_object["l1_cache"]["policy"];
    l1_cache_memory_budget_mb_ = json_object["l1_cache"]["cache_memory_budget_mb"];
    l2_cache_policy_ = json_object["l2_cache"]["policy"];
    l2_cache_memory_budget_mb_ = json_object["l2_cache"]["cache_memory_budget_mb"];
    persistent_table_path_ = json_object["kv_store"]["persistent_table"]["path"];
    persistent_table_phisical_block_size_ =
        json_object["kv_store"]["persistent_table"]["physical_block_size"];

    optimizer_type_ = json_object["optimizer"]["type"];
    if (optimizer_type_ == "sgd") {
      line_size_ = embedding_size_;
    } else if (optimizer_type_ == "momentum") {
      beta_ = json_object["optimizer"]["beta"];
      line_size_ = embedding_size_ * 2;
    } else if (optimizer_type_ == "adam") {
      beta1_ = json_object["optimizer"]["beta1"];
      beta2_ = json_object["optimizer"]["beta2"];
      epsilon_ = json_object["optimizer"]["epsilon"];
      do_bias_correction_ = json_object["optimizer"]["do_bias_correction"];
      line_size_ = embedding_size_ * 3;
    } else {
      UNIMPLEMENTED();
    }
    auto learning_rate_schedule_object = json_object["learning_rate_schedule"];
    learning_rate_ = learning_rate_schedule_object["learning_rate"];
    warmup_type_ = learning_rate_schedule_object["warmup"]["type"];
    if (warmup_type_ == "linear") {
      warmup_conf_.mutable_linear_conf()->set_warmup_batches(
          learning_rate_schedule_object["warmup"]["warmup_batches"]);
      warmup_conf_.mutable_linear_conf()->set_start_multiplier(
          learning_rate_schedule_object["warmup"]["start_multiplier"]);
    } else if (warmup_type_ == "constant") {
      warmup_conf_.mutable_constant_conf()->set_warmup_batches(
          learning_rate_schedule_object["warmup"]["warmup_batches"]);
      warmup_conf_.mutable_constant_conf()->set_multiplier(
          learning_rate_schedule_object["warmup"]["multiplier"]);
    } else if (warmup_type_ == "none") {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    learning_rate_decay_type_ = learning_rate_schedule_object["learning_rate_decay"]["type"];
    if (learning_rate_decay_type_ == "polynomial") {
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_decay_batches(
          learning_rate_schedule_object["learning_rate_decay"]["decay_batches"]);
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_end_learning_rate(
          learning_rate_schedule_object["learning_rate_decay"]["end_learning_rate"]);
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_power(
          learning_rate_schedule_object["learning_rate_decay"]["power"]);
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_cycle(
          learning_rate_schedule_object["learning_rate_decay"]["cycle"]);
    } else if (learning_rate_decay_type_ == "none") {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
  }
  ~EmbeddingOptions() = default;

  std::string EmbeddingName() const { return embedding_name_; }
  int64_t EmbeddingSize() const { return embedding_size_; }
  int64_t LineSize() const { return line_size_; }
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

 private:
  std::string embedding_name_;
  int64_t embedding_size_;
  int64_t line_size_;
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
};

}  // namespace embedding
}  // namespace oneflow
#endif  // ONEFLOW_EMBEDDING_EMBEDDING_OPTIONS_H_
