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

#include "oneflow/core/embedding/key_value_store.h"
#include "oneflow/core/embedding/key_value_store_options.h"

namespace oneflow {

namespace embedding {

inline bool UseDynamicMemoryAllocation() {
  bool use_dynamic_memory_allocation =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION", true);
#if CUDA_VERSION >= 11020
  return use_dynamic_memory_allocation;
#else
  if (use_dynamic_memory_allocation) {
    LOG(WARNING)
        << "Dynamic memory allocation only support when cuda_version greater equal than 11200. ";
  }
  return false;
#endif
}

#ifdef WITH_CUDA

class ValuesPtr {
 public:
  ValuesPtr() {
    lookup_num_unique_ = 0;
    num_unique_counter_ = -1;

    has_lookup_values_ = false;
    lookup_values_counter_ = -1;
    has_lookup_embeddings_ = false;
    lookup_embeddings_counter_ = -1;
    updated_values_counter_ = -1;
  }
  ~ValuesPtr() {
    if (has_lookup_values_) { OF_CUDA_CHECK(cudaFree(lookup_values_)); }
    if (has_lookup_embeddings_) { OF_CUDA_CHECK(cudaFree(lookup_embeddings_)); }
  }
  uint32_t GetNumUnique(int iter) {
    CHECK_EQ(iter, num_unique_counter_);
    return lookup_num_unique_;
  }
  void SetNumUnique(uint32_t num_unique, int iter) {
    lookup_num_unique_ = num_unique;
    num_unique_counter_ = iter;
  }
  void* GetLookupValuesPtr(int iter) {
    CHECK(has_lookup_values_);
    CHECK_EQ(lookup_values_counter_, iter);
    return lookup_values_;
  }
  void* GetLookupEmbeddingsPtr(int iter) {
    CHECK(has_lookup_embeddings_);
    CHECK_EQ(lookup_embeddings_counter_, iter);
    return lookup_embeddings_;
  }
  void* MallocLookupValuesPtr(size_t data_size, cudaStream_t cuda_stream) {
    if (has_lookup_values_ && lookup_values_size_ >= data_size) {
      // do nothing
    } else {
      if (has_lookup_values_) { OF_CUDA_CHECK(cudaFreeAsync(lookup_values_, cuda_stream)); }
      OF_CUDA_CHECK(cudaMallocAsync(&(lookup_values_), data_size, cuda_stream));
      has_lookup_values_ = true;
      lookup_values_size_ = data_size;
    }
    lookup_values_counter_++;
    return lookup_values_;
  }
  bool HasLookupEmbeddings() { return has_lookup_embeddings_; }
  void* MallocLookupEmbeddingsPtr(size_t data_size, cudaStream_t cuda_stream) {
    if (has_lookup_embeddings_ && lookup_embeddings_size_ >= data_size) {
      // do nothing
    } else {
      if (has_lookup_embeddings_) { OF_CUDA_CHECK(cudaFreeAsync(lookup_embeddings_, cuda_stream)); }
      OF_CUDA_CHECK(cudaMallocAsync(&lookup_embeddings_, data_size, cuda_stream));
      has_lookup_embeddings_ = true;
      lookup_embeddings_size_ = data_size;
    }
    lookup_embeddings_counter_++;
    return lookup_embeddings_;
  }

  void* MallocUpdatedValuesPtr(size_t data_size, cudaStream_t cuda_stream) {
    CHECK(!has_updated_values_);
    OF_CUDA_CHECK(cudaMallocAsync(&updated_values_, data_size, cuda_stream));
    has_updated_values_ = true;
    updated_values_counter_++;
    return updated_values_;
  }
  void* GetUpdatedValuesPtr(int iter) {
    CHECK(has_updated_values_);
    CHECK_EQ(updated_values_counter_, iter);
    return updated_values_;
  }
  void FreeUpdatedValuesPtr(cudaStream_t cuda_stream) {
    CHECK(has_updated_values_);
    OF_CUDA_CHECK(cudaFreeAsync(updated_values_, cuda_stream));
    has_updated_values_ = false;
  }

 private:
  uint32_t lookup_num_unique_;
  int num_unique_counter_;

  void* lookup_values_;
  bool has_lookup_values_;
  size_t lookup_values_size_;
  int lookup_values_counter_;

  void* lookup_embeddings_;
  bool has_lookup_embeddings_;
  size_t lookup_embeddings_size_;
  int lookup_embeddings_counter_;

  void* updated_values_;
  bool has_updated_values_;
  int updated_values_counter_;
};

class EmbeddingManager final {
 public:
  EmbeddingManager() = default;
  ~EmbeddingManager() = default;

  void SaveSnapshot(const std::string& embedding_name, int64_t local_rank_id, int64_t rank_id,
                    const std::string& snapshot_name);
  void LoadSnapshot(const std::string& embedding_name, int64_t local_rank_id, int64_t rank_id,
                    const std::string& snapshot_name);
  ValuesPtr* GetValuesPtr(const std::string& embedding_name, int64_t rank_id);

  KeyValueStore* GetKeyValueStore(const std::string& embedding_name, int64_t rank_id);

  void CreateKeyValueStore(const KeyValueStoreOptions& options, int64_t local_rank_id,
                           int64_t rank_id, int64_t world_size);

 private:
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<KeyValueStore>> key_value_store_map_;
  std::mutex mutex_;
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<ValuesPtr>> values_ptrs_map_;
};

#endif  // WITH_CUDA

}  // namespace embedding
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_EMBEDDING_MANAGER_H_
