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

#ifdef WITH_CUDA

inline bool UseDynamicMemoryAllocation() {
  static bool use_dynamic_memory_allocation =
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

constexpr int64_t kRingBufferSize = 8;

struct NumUniqueState {
  uint32_t num_unique;
  std::vector<uint32_t> num_unique_matrix;
  int64_t iter;
};

class EmbeddingState {
 public:
  EmbeddingState() = default;
  virtual ~EmbeddingState() = default;

  virtual void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual void* LookupOutValues(int64_t iter) = 0;
  virtual void* LookupOutEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingLookupEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* EmbeddingShuffleInEmbeddings(int64_t iter) = 0;
  virtual void OnEmbeddingShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingGradientShuffleStart(user_op::KernelComputeContext* ctx,
                                               int64_t iter) = 0;
  virtual void OnEmbeddingGradientShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingUpdateStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* UpdateInValues(int64_t iter) = 0;
  virtual void* UpdateOutValues(int64_t iter) = 0;
  virtual void OnEmbeddingUpdateEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void OnEmbeddingPutStart(user_op::KernelComputeContext* ctx, int64_t iter) = 0;
  virtual const void* PutInValues(int64_t iter) = 0;
  virtual void OnEmbeddingPutEnd(user_op::KernelComputeContext* ctx, int64_t iter) = 0;

  virtual void AllocTmpBuffer(user_op::KernelComputeContext* ctx, void** ptr, size_t size) = 0;
  virtual void FreeTmpBuffer(user_op::KernelComputeContext* ctx, void* ptr) = 0;

  virtual void SetNumUniqueState(uint32_t num_unique,
                                 const std::vector<uint32_t>& num_unique_matrix, int64_t iter) = 0;

  virtual uint32_t GetNumUnique(int64_t iter) = 0;

  virtual const std::vector<uint32_t>& GetNumUniqueMatrix(int64_t iter) = 0;
};

class DynamicAllocationEmbeddingState final : public EmbeddingState {
 public:
  DynamicAllocationEmbeddingState() = default;
  ~DynamicAllocationEmbeddingState() {}

  void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    uint32_t num_unique = this->GetNumUnique(iter);
    size_t lookup_values_size =
        GetCudaAlignedSize(num_unique * line_size * GetSizeOfDataType(unique_values->data_type()));
    if (lookup_values_size_ < lookup_values_size) {
      if (has_lookup_values_) { OF_CUDA_CHECK(cudaFreeAsync(lookup_values_, cuda_stream)); }
      OF_CUDA_CHECK(cudaMallocAsync(&lookup_values_, lookup_values_size, cuda_stream));
      has_lookup_values_ = true;
      lookup_values_size_ = lookup_values_size;
      if (ctx->has_output("embeddings", 0)) {
        user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
        const size_t lookup_embeddings_size = GetCudaAlignedSize(
            num_unique * embedding_size * GetSizeOfDataType(embeddings->data_type()));
        if (lookup_embeddings_size_ < lookup_values_size) {
          if (has_lookup_embeddings_) {
            OF_CUDA_CHECK(cudaFreeAsync(lookup_embeddings_, cuda_stream));
          }
          OF_CUDA_CHECK(cudaMallocAsync(&lookup_embeddings_, lookup_embeddings_size, cuda_stream));
          has_lookup_embeddings_ = true;
          lookup_embeddings_size_ = lookup_embeddings_size;
        }
      } else {
        lookup_embeddings_ = nullptr;
      }
    }
  }

  void* LookupOutValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_values_);
    return lookup_values_;
  }

  void* LookupOutEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_embeddings_);
    return lookup_embeddings_;
  }

  void OnEmbeddingLookupEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  void OnEmbeddingShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  const void* EmbeddingShuffleInEmbeddings(int64_t iter) override {
    if (has_lookup_embeddings_) {
      return lookup_embeddings_;
    } else {
      CHECK(has_lookup_values_);
      return lookup_values_;
    }
  }

  void OnEmbeddingShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  void OnEmbeddingGradientShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) override {}
  void OnEmbeddingGradientShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {}

  void OnEmbeddingUpdateStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    uint32_t num_unique = this->GetNumUnique(iter);
    size_t update_values_size = GetCudaAlignedSize(
        num_unique * line_size * GetSizeOfDataType(updated_unique_embeddings->data_type()));
    OF_CUDA_CHECK(cudaMallocAsync(&updated_values_, update_values_size,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }

  const void* UpdateInValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_values_);
    return lookup_values_;
  }

  void* UpdateOutValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return updated_values_;
  }

  void OnEmbeddingUpdateEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }
  void OnEmbeddingPutStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  const void* PutInValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return updated_values_;
  }

  void OnEmbeddingPutEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    cudaFreeAsync(updated_values_, ctx->stream()->As<ep::CudaStream>()->cuda_stream());
  }

  void AllocTmpBuffer(user_op::KernelComputeContext* ctx, void** ptr, size_t size) override {}
  void FreeTmpBuffer(user_op::KernelComputeContext* ctx, void* ptr) override {}

  void SetNumUniqueState(uint32_t num_unique, const std::vector<uint32_t>& num_unique_matrix,
                         int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    num_unique_states_.at(index).num_unique = num_unique;
    num_unique_states_.at(index).num_unique_matrix = num_unique_matrix;
    num_unique_states_.at(index).iter = iter;
  }

  uint32_t GetNumUnique(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const NumUniqueState& state = num_unique_states_.at(index);
    CHECK_EQ(state.iter, iter) << "saved iter: " << state.iter << " current iter: " << iter;
    return state.num_unique;
  }

  const std::vector<uint32_t>& GetNumUniqueMatrix(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const NumUniqueState& state = num_unique_states_.at(index);
    CHECK_EQ(state.iter, iter) << "saved iter: " << state.iter << " current iter: " << iter;
    return state.num_unique_matrix;
  }

 private:
  void* lookup_values_;
  size_t lookup_values_size_;
  bool has_lookup_values_;
  void* lookup_embeddings_;
  size_t lookup_embeddings_size_;
  bool has_lookup_embeddings_;
  void* updated_values_;
  uint64_t iter_;
  std::vector<NumUniqueState> num_unique_states_;
  std::mutex mutex_;
};

class StaticAllocationEmbeddingState final : public EmbeddingState {
 public:
  StaticAllocationEmbeddingState() = default;
  ~StaticAllocationEmbeddingState() override = default;

  void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    lookup_out_values_ = unique_values->mut_dptr();
    if (ctx->has_output("embeddings", 0)) {
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      has_lookup_out_embeddings_ = true;
      lookup_out_embeddings_ = embeddings->mut_dptr();
    }
  }

  void* LookupOutValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return lookup_out_values_;
  }

  void* LookupOutEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_out_embeddings_);
    return lookup_out_embeddings_;
  }

  void OnEmbeddingLookupEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    lookup_out_values_ = nullptr;
    lookup_out_embeddings_ = nullptr;
    has_lookup_out_embeddings_ = false;
  }

  void OnEmbeddingShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* cur_rank_embeddings =
        ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0);
    embedding_shuffle_in_embeddings_ = cur_rank_embeddings->dptr();
  }

  const void* EmbeddingShuffleInEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return embedding_shuffle_in_embeddings_;
  }

  void OnEmbeddingShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    embedding_shuffle_in_embeddings_ = nullptr;
  }

  void OnEmbeddingGradientShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) override {}

  void OnEmbeddingGradientShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {}

  void OnEmbeddingUpdateStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    update_in_values_ = unique_embeddings->dptr();
    update_out_values_ = updated_unique_embeddings->mut_dptr();
  }

  const void* UpdateInValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return update_in_values_;
  }

  void* UpdateOutValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return update_out_values_;
  }

  void OnEmbeddingUpdateEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    update_in_values_ = nullptr;
    update_out_values_ = nullptr;
  }

  void OnEmbeddingPutStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    put_in_values_ = unique_embeddings->dptr();
  }

  const void* PutInValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return put_in_values_;
  }

  void OnEmbeddingPutEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    put_in_values_ = nullptr;
  }

  void AllocTmpBuffer(user_op::KernelComputeContext* ctx, void** ptr, size_t size) override {}
  void FreeTmpBuffer(user_op::KernelComputeContext* ctx, void* ptr) override {}

  void SetNumUniqueState(uint32_t num_unique, const std::vector<uint32_t>& num_unique_matrix,
                         int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    num_unique_states_.at(index).num_unique = num_unique;
    num_unique_states_.at(index).num_unique_matrix = num_unique_matrix;
    num_unique_states_.at(index).iter = iter;
  }

  uint32_t GetNumUnique(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const NumUniqueState& state = num_unique_states_.at(index);
    CHECK_EQ(state.iter, iter) << "saved iter: " << state.iter << " current iter: " << iter;
    return state.num_unique;
  }

  const std::vector<uint32_t>& GetNumUniqueMatrix(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const NumUniqueState& state = num_unique_states_.at(index);
    CHECK_EQ(state.iter, iter) << "saved iter: " << state.iter << " current iter: " << iter;
    return state.num_unique_matrix;
  }

  void* lookup_out_values_;
  void* lookup_out_embeddings_;
  bool has_lookup_out_embeddings_;
  const void* embedding_shuffle_in_embeddings_;
  const void* update_in_values_;
  void* update_out_values_;
  const void* put_in_values_;
  int64_t iter_;
  std::vector<NumUniqueState> num_unique_states_;
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
