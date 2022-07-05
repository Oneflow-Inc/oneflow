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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace embedding {

inline bool UseDynamicMemoryAllocation() {
  static bool use_dynamic_memory_allocation =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION", false);
#if CUDA_VERSION >= 11020
  return use_dynamic_memory_allocation;
#else
  if (use_dynamic_memory_allocation) {
    LOG(WARNING)
        << "Dynamic memory allocation only support when cuda_version greater equal than 11.2. ";
  }
  return false;
#endif
}

#ifdef WITH_CUDA

class EmbeddingState {
 public:
  EmbeddingState() = default;
  virtual ~EmbeddingState() = default;

  virtual void OnEmbeddingPrefetchStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual void OnEmbeddingPrefetchEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual void* LookupUniqueValues(int64_t iter) = 0;
  virtual void* LookupEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingLookupEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* EmbeddingShuffleCurRankEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingGradientShuffleStart(user_op::KernelComputeContext* ctx,
                                               int64_t iter) = 0;
  virtual void OnEmbeddingGradientShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingUpdateStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* EmbeddingUpdateUniqueEmbeddings(int64_t iter) = 0;
  virtual void* EmbeddingUpdateUpdatedUniqueEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingUpdateEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingPutStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* EmbeddingPutUniqueEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingPutEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingFusedUpdatePutStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* EmbeddingFusedUpdatePutUniqueEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingFusedUpdatePutEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void AllocTmpBuffer(user_op::KernelComputeContext* ctx, void** ptr, size_t size) = 0;
  virtual void FreeTmpBuffer(user_op::KernelComputeContext* ctx, void* ptr) = 0;

  virtual void SetIdFinalNumUnique(uint32_t final_num_unique, int64_t iter) = 0;
  virtual void SetIdNumUniqueMatrix(const std::vector<uint32_t>& num_unique_matrix,
                                    int64_t iter) = 0;
  virtual uint32_t GetIdNumUnique(int64_t iter) = 0;
  virtual const std::vector<uint32_t>& GetIdNumUniqueMatrix(int64_t iter) = 0;
};

class EmbeddingManager final {
 public:
  EmbeddingManager() = default;
  ~EmbeddingManager() = default;

  void SaveSnapshot(const std::string& embedding_name, int64_t local_rank_id, int64_t rank_id,
                    const std::string& snapshot_name);
  void LoadSnapshot(const std::string& embedding_name, int64_t local_rank_id, int64_t rank_id,
                    const std::string& snapshot_name);

  KeyValueStore* GetKeyValueStore(const std::string& embedding_name, int64_t rank_id);
  EmbeddingState* GetEmbeddingState(const std::string& embedding_name, int64_t rank_id);
  void CreateKeyValueStore(const KeyValueStoreOptions& options, int64_t local_rank_id,
                           int64_t rank_id, int64_t world_size);

 private:
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<KeyValueStore>> key_value_store_map_;
  HashMap<std::pair<std::string, int64_t>, std::unique_ptr<EmbeddingState>> embedding_state_map_;
  std::mutex mutex_;
};

#endif  // WITH_CUDA

}  // namespace embedding
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_EMBEDDING_MANAGER_H_
