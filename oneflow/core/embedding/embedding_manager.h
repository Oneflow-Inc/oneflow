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

constexpr int64_t kRingBufferSize = 8;

class NumUniques {
 public:
  NumUniques() {
    num_unique_.resize(kRingBufferSize);
    num_unique_iter_.resize(kRingBufferSize);
    num_unique_matrix_.resize(kRingBufferSize);
    num_unique_matrix_iter_.resize(kRingBufferSize);
  }
  ~NumUniques() = default;

  uint32_t GetNumUnique(int64_t iter);

  void SetNumUnique(uint32_t num_unique, int64_t iter);

  void SetNumUniqueMatrix(const std::vector<uint32_t>& num_unique_matrix, int64_t iter);

  const std::vector<uint32_t>& GetNumUniqueMatrix(int64_t iter);

 private:
  std::vector<uint32_t> num_unique_;
  std::vector<int64_t> num_unique_iter_;
  std::vector<std::vector<uint32_t>> num_unique_matrix_;
  std::vector<int64_t> num_unique_matrix_iter_;
  std::mutex mutex_;
};

class ValuesPtr {
 public:
  ValuesPtr() {
    has_lookup_values_ = false;
    has_lookup_embeddings_ = false;
    has_updated_values_ = false;
  }
  ~ValuesPtr() {
    if (has_lookup_values_) { OF_CUDA_CHECK(cudaFree(lookup_values_)); }
    if (has_lookup_embeddings_) { OF_CUDA_CHECK(cudaFree(lookup_embeddings_)); }
  }
  void* GetLookupValuesPtr(int64_t iter);

  void* GetLookupEmbeddingsPtr(int64_t iter);

  void* MallocLookupValuesPtr(int64_t iter, size_t data_size, cudaStream_t cuda_stream);

  bool HasLookupEmbeddings();

  void* MallocLookupEmbeddingsPtr(int64_t iter, size_t data_size, cudaStream_t cuda_stream);

  void* MallocUpdatedValuesPtr(int64_t iter, size_t data_size, cudaStream_t cuda_stream);

  void* GetUpdatedValuesPtr(int64_t iter);

  void FreeUpdatedValuesPtr(int64_t iter, cudaStream_t cuda_stream);

 private:
  void* lookup_values_;
  bool has_lookup_values_;
  size_t lookup_values_size_;
  int64_t lookup_values_iter_;

  void* lookup_embeddings_;
  bool has_lookup_embeddings_;
  size_t lookup_embeddings_size_;
  int64_t lookup_embeddings_iter_;

  void* updated_values_;
  bool has_updated_values_;
  int64_t updated_values_iter_;
  std::mutex mutex_;
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

  NumUniques* GetNumUniques(const std::string& embedding_name, int64_t rank_id);

  KeyValueStore* GetKeyValueStore(const std::string& embedding_name, int64_t rank_id);

  void CreateKeyValueStore(const KeyValueStoreOptions& options, int64_t local_rank_id,
                           int64_t rank_id, int64_t world_size);

 private:
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<KeyValueStore>> key_value_store_map_;
  std::mutex mutex_;
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<ValuesPtr>> values_ptrs_map_;
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<NumUniques>> num_uniques_map_;
};

#endif  // WITH_CUDA

}  // namespace embedding
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_EMBEDDING_MANAGER_H_
