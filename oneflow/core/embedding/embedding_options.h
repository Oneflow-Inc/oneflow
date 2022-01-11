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
    embedding_name_ = "EmbeddingTest";  // json_object["embedding_name"];
    // key_type_ = json_object["key_type"];
    // value_type_ = json_object["value_type"];
    embedding_size_ = ParseIntegerFromEnv("EMBEDDING_SIZE", 128);  // json_object["embedding_size"];
    // l1_cache_policy_ = json_object["l1_cache_policy"];
    // l2_cache_policy_ = json_object["l2_cache_policy"];
    cache_memory_budget_mb_ = ParseIntegerFromEnv("CACHE_MEMORY_BUDGET_MB", 0);
    kv_store_ = GetStringFromEnv("KEY_VALUE_STORE", "");  // json_object["kv_store"];
    fixed_table_path_ =
        GetStringFromEnv("BLOCK_BASED_PATH", "");  // json_object["fixed_table_path"];
    fixed_table_block_size_ = 512;                 // json_object["fixed_table_block_size"];
    // fixed_table_chunk_size_ = json_object["fixed_table_chunk_size"];
    num_keys_ = ParseIntegerFromEnv("NUM_KEYS", 0);  // json_object["num_keys"];
    num_device_keys_ =
        ParseIntegerFromEnv("NUM_DEVICE_KEYS", 0);  // json_object["num_device_keys"];
    optimizer_ = json_object["optimizer"];
    base_learning_rate_ = json_object["base_learning_rate"];
    if (optimizer_ == "sgd") {
      // do nothing
    } else if (optimizer_ == "momentum") {
      beta_ = json_object["optimizer_conf"]["beta"];
    } else if (optimizer_ == "adam") {
      beta1_ = json_object["optimizer_conf"]["beta1"];
      beta2_ = json_object["optimizer_conf"]["beta2"];
      epsilon_ = json_object["optimizer_conf"]["epsilon"];
      amsgrad_ = json_object["optimizer_conf"]["amsgrad"];
    } else {
      UNIMPLEMENTED();
    }
    warmup_type_ = json_object["warmup_type"];
    if (warmup_type_ == "linear") {
      warmup_conf_.mutable_linear_conf()->set_warmup_batches(
          json_object["warmup_conf"]["warmup_batches"]);
      warmup_conf_.mutable_linear_conf()->set_start_multiplier(
          json_object["warmup_conf"]["start_multiplier"]);
    } else if (warmup_type_ == "constant") {
      warmup_conf_.mutable_constant_conf()->set_warmup_batches(
          json_object["warmup_conf"]["warmup_batches"]);
      warmup_conf_.mutable_constant_conf()->set_multiplier(
          json_object["warmup_conf"]["multiplier"]);
    } else if (warmup_type_ == "none") {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    learning_rate_decay_type_ = json_object["learning_rate_decay_type"];
    if (learning_rate_decay_type_ == "polynomial") {
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_decay_batches(
          json_object["learning_rate_decay_conf"]["decay_batches"]);
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_end_learning_rate(
          json_object["learning_rate_decay_conf"]["end_learning_rate"]);
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_power(
          json_object["learning_rate_decay_conf"]["power"]);
      learning_rate_decay_conf_.mutable_polynomial_conf()->set_cycle(
          json_object["learning_rate_decay_conf"]["cycle"]);
    } else if (learning_rate_decay_type_ == "none") {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
  }
  ~EmbeddingOptions() = default;

  std::string EmbeddingName() const { return embedding_name_; }
  DataType KeyType() const { return key_type_; }
  DataType ValueType() const { return value_type_; }
  int64_t EmbeddingSize() const { return embedding_size_; }
  std::string L1CachePolicy() const { return l1_cache_policy_; }
  std::string L2CachePolicy() const { return l2_cache_policy_; }
  std::string KVStore() const { return kv_store_; }
  std::string FixedTablePath() const { return fixed_table_path_; }
  int64_t CacheMemoryBudgetMb() const { return cache_memory_budget_mb_; }
  int64_t FixedTableBlockSize() const { return fixed_table_block_size_; }
  int64_t FixedTableChunkSize() const { return fixed_table_chunk_size_; }
  int64_t NumKeys() const { return num_keys_; }
  int64_t NumDeviceKeys() const { return num_device_keys_; }
  std::string Optimizer() const { return optimizer_; }
  float BaseLearningRate() const { return base_learning_rate_; }
  std::string WarmupType() const { return warmup_type_; }
  WarmupConf WarmupConfProto() const { return warmup_conf_; }
  std::string LearningRateDecayType() const { return learning_rate_decay_type_; }
  LearningRateDecayConf LearningRateDecayConfProto() const { return learning_rate_decay_conf_; }

 private:
  std::string embedding_name_;
  DataType key_type_;
  DataType value_type_;
  int64_t embedding_size_;
  std::string l1_cache_policy_;
  std::string l2_cache_policy_;
  std::string kv_store_;
  std::string fixed_table_path_;
  int64_t cache_memory_budget_mb_;
  int64_t fixed_table_block_size_;
  int64_t fixed_table_chunk_size_;
  int64_t num_keys_;
  int64_t num_device_keys_;
  std::string optimizer_;
  float base_learning_rate_;
  float beta_;
  float beta1_;
  float beta2_;
  float epsilon_;
  float amsgrad_;
  std::string warmup_type_;
  WarmupConf warmup_conf_;
  std::string learning_rate_decay_type_;
  LearningRateDecayConf learning_rate_decay_conf_;
};

}  // namespace embedding
}  // namespace oneflow
#endif  // ONEFLOW_EMBEDDING_EMBEDDING_OPTIONS_H_
