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

inline bool UseEmbeddingShuffleP2PKernel(DataType embedding_dtype, DataType idx_dtype) {
  static bool use_embedding_shuffle_p2p_env =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_USE_P2P", false);
  static bool add_id_shuffle_copy_out_env =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ADD_ID_SHUFFLE_COPY_OUT", true);
  static bool enable_quantized_comm =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM", false);
  if (use_embedding_shuffle_p2p_env) {
    if (embedding_dtype != DataType::kFloat16 || idx_dtype != DataType::kUInt32) {
      // p2p kernel only registered kFloat16 and kUint32.
      return false;
    }
    if (!add_id_shuffle_copy_out_env) {
      // when not enable id shuffle copy out, the ptrs change every iter.
      return false;
    }
    if (enable_quantized_comm) {
      // p2p kernel not support quantize comm.
      return false;
    }
    if (UseDynamicMemoryAllocation()) {
      // p2p kernel not support dynamic memory allocation.
      return false;
    }
  }
#if CUDA_VERSION >= 11030
  return use_embedding_shuffle_p2p_env;
#else
  if (use_embedding_shuffle_p2p_env) {
    LOG(WARNING)
        << "embedding shuffle p2p kernel only support when cuda_version greater equal than 11.3. ";
  }
  return false;
#endif
}

inline bool UseEmbeddingGradientShuffleP2PKernel(DataType embedding_dtype, DataType idx_dtype) {
  static bool use_embedding_gradient_shuffle_p2p_env =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_EMBEDDING_GRADIENT_SHUFFLE_USE_P2P", false);
  static bool add_id_shuffle_copy_out_env =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ADD_ID_SHUFFLE_COPY_OUT", true);
  static bool enable_quantized_comm =
      ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM", false);
  if (use_embedding_gradient_shuffle_p2p_env) {
    if (embedding_dtype != DataType::kFloat16 || idx_dtype != DataType::kUInt32) {
      // p2p kernel only registered kFloat16 and kUint32.
      return false;
    }
    if (!add_id_shuffle_copy_out_env) {
      // when not enable id shuffle copy out, the ptrs change every iter.
      return false;
    }
    if (enable_quantized_comm) {
      // p2p kernel not support quantize comm.
      return false;
    }
    if (UseDynamicMemoryAllocation()) {
      // p2p kernel not support dynamic memory allocation.
      return false;
    }
  }
#if CUDA_VERSION >= 11030
  return use_embedding_gradient_shuffle_p2p_env;
#else
  if (use_embedding_gradient_shuffle_p2p_env) {
    LOG(WARNING) << "embedding gradient shuffle p2p kernel only support when cuda_version greater "
                    "equal than 11.3. ";
  }
  return false;
#endif
}

#ifdef WITH_CUDA

class TmpBufferAllocator {
 public:
  TmpBufferAllocator() = default;
  virtual ~TmpBufferAllocator() = default;

  virtual void Allocate(void** ptr, size_t size) = 0;
  virtual void Free(void* ptr) = 0;
};

class EmbeddingState {
 public:
  EmbeddingState() = default;
  virtual ~EmbeddingState() = default;

  virtual std::unique_ptr<TmpBufferAllocator> NewTmpBufferAllocator(
      user_op::KernelComputeContext* ctx) = 0;

  virtual void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual void* LookupUniqueValues(int64_t iter) = 0;
  virtual void* LookupEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingLookupEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingGatherStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* EmbeddingGatherIn(int64_t iter) = 0;
  virtual void OnEmbeddingGatherEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* EmbeddingShuffleCurRankEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

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
