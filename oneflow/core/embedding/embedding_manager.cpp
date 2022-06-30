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
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/embedding/persistent_table_key_value_store.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/embedding/cached_key_value_store.h"

namespace oneflow {

namespace embedding {

#ifdef WITH_CUDA

constexpr size_t kDefaultMaxQueryLength = 65536;

class DynamicAllocationEmbeddingState final : public EmbeddingState {
 public:
  DynamicAllocationEmbeddingState() {
    num_unique_states_.resize(kRingBufferSize);
    has_lookup_values_ = false;
    has_lookup_embeddings_ = false;
    // TODO: set mem pool, use cudaMallocFromPoolAsync
  }
  ~DynamicAllocationEmbeddingState() {
    if (has_lookup_values_) { OF_CUDA_CHECK(cudaFree(lookup_values_)); }
    if (has_lookup_embeddings_) { OF_CUDA_CHECK(cudaFree(lookup_embeddings_)); }
  }

  void OnEmbeddingPrefetchStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  void OnEmbeddingPrefetchEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing, check ptrs is freed
  }

  void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    iter_ = iter;
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    uint32_t num_unique = this->GetNumUnique(iter);
    size_t lookup_values_size =
        GetCudaAlignedSize(num_unique * line_size * GetSizeOfDataType(unique_values->data_type()));
    if (lookup_values_size_ < lookup_values_size) {
      if (has_lookup_values_) { OF_CUDA_CHECK(cudaFreeAsync(lookup_values_, cuda_stream)); }
      // cudaMallocFromPoolAsync
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
    OF_CUDA_CHECK(
        cudaFreeAsync(updated_values_, ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }

  void AllocTmpBuffer(user_op::KernelComputeContext* ctx, void** ptr, size_t size) override {
    OF_CUDA_CHECK(cudaMallocAsync(ptr, size, ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }

  void FreeTmpBuffer(user_op::KernelComputeContext* ctx, void* ptr) override {
    OF_CUDA_CHECK(cudaFreeAsync(ptr, ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }

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
  StaticAllocationEmbeddingState() { num_unique_states_.resize(kRingBufferSize); }
  ~StaticAllocationEmbeddingState() override = default;

  void OnEmbeddingPrefetchStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    tmp_buffer_ptr_ = tmp_buffer->mut_dptr();
    tmp_buffer_offset_ = 0;
    tmp_buffer_size_ = tmp_buffer->shape_view().elem_cnt();
  }

  void OnEmbeddingPrefetchEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    tmp_buffer_ptr_ = nullptr;
    tmp_buffer_offset_ = 0;
    tmp_buffer_size_ = 0;
  }

  void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    lookup_out_values_ = unique_values->mut_dptr();
    if (ctx->has_output("embeddings", 0)) {
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      has_lookup_out_embeddings_ = true;
      lookup_out_embeddings_ = embeddings->mut_dptr();
    }
  }

  void* LookupOutValues(int64_t iter) override { return lookup_out_values_; }

  void* LookupOutEmbeddings(int64_t iter) override {
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

  const void* UpdateInValues(int64_t iter) override { return update_in_values_; }

  void* UpdateOutValues(int64_t iter) override { return update_out_values_; }

  void OnEmbeddingUpdateEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    update_in_values_ = nullptr;
    update_out_values_ = nullptr;
  }

  void OnEmbeddingPutStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    put_in_values_ = unique_embeddings->dptr();
  }

  const void* PutInValues(int64_t iter) override { return put_in_values_; }

  void OnEmbeddingPutEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    put_in_values_ = nullptr;
  }

  void AllocTmpBuffer(user_op::KernelComputeContext* ctx, void** ptr, size_t size) override {
    CHECK(tmp_buffer_ptr_ != nullptr);
    CHECK_GE(tmp_buffer_offset_, 0);
    CHECK_LE(tmp_buffer_offset_ + size, tmp_buffer_size_);
    *ptr = reinterpret_cast<char*>(tmp_buffer_ptr_) + tmp_buffer_offset_;
    tmp_buffer_offset_ += size;
  }
  void FreeTmpBuffer(user_op::KernelComputeContext* ctx, void* ptr) override {
    // do nothing, can not get the size
  }

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
  std::vector<NumUniqueState> num_unique_states_;
  void* tmp_buffer_ptr_ = nullptr;
  int64_t tmp_buffer_offset_ = 0;
  size_t tmp_buffer_size_ = 0;
  std::mutex mutex_;
};

EmbeddingState* EmbeddingManager::GetEmbeddingState(const std::string& embedding_name,
                                                    int64_t rank_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, rank_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = embedding_state_map_.find(map_key);
  // for id shuffle test, not need to create table
  if (it == embedding_state_map_.end()) {
    LOG(WARNING) << "create embedding state: " << embedding_name << "-" << rank_id;
    if (UseDynamicMemoryAllocation()) {
      it =
          embedding_state_map_.emplace(map_key, std::make_unique<DynamicAllocationEmbeddingState>())
              .first;
    } else {
      it = embedding_state_map_.emplace(map_key, std::make_unique<StaticAllocationEmbeddingState>())
               .first;
    }
  }
  return it->second.get();
}

KeyValueStore* EmbeddingManager::GetKeyValueStore(const std::string& embedding_name,
                                                  int64_t rank_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, rank_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = key_value_store_map_.find(map_key);
  CHECK(it != key_value_store_map_.end())
      << "Can not find embedding: " << embedding_name << "-" << rank_id;
  return it->second.get();
}

void EmbeddingManager::CreateKeyValueStore(const KeyValueStoreOptions& key_value_store_options,
                                           int64_t local_rank_id, int64_t rank_id,
                                           int64_t world_size) {
  CudaCurrentDeviceGuard guard(local_rank_id);
  const std::string& name = key_value_store_options.Name();
  const uint32_t line_size = key_value_store_options.LineSize();
  std::pair<std::string, int64_t> map_key = std::make_pair(name, rank_id);
  std::unique_lock<std::mutex> lock(mutex_);

  std::unique_ptr<KeyValueStore> store;
  PersistentTableKeyValueStoreOptions options{};
  const std::vector<std::string>& persistent_table_paths =
      key_value_store_options.PersistentTablePaths();
  CHECK_EQ(persistent_table_paths.size(), world_size);
  options.table_options.path = persistent_table_paths.at(rank_id);
  options.table_options.value_size = line_size * key_value_store_options.ValueTypeSize();
  options.table_options.key_size = key_value_store_options.KeyTypeSize();
  options.table_options.physical_block_size =
      key_value_store_options.PersistentTablePhysicalBlockSize();
  options.table_options.target_chunk_size_mb = 4 * 1024;
  options.table_options.capacity_hint = key_value_store_options.PersistentTableCapacityHint();
  store = NewPersistentTableKeyValueStore(options);
  const std::vector<CacheOptions>& cache_options = key_value_store_options.GetCachesOptions();
  for (int i = cache_options.size() - 1; i >= 0; --i) {
    std::unique_ptr<Cache> cache = NewCache(cache_options.at(i));
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }
  store->ReserveQueryLength(kDefaultMaxQueryLength);
  CHECK(key_value_store_map_.emplace(map_key, std::move(store)).second)
      << "Can't create an embedding with same name of an existing embedding, the name: " << name;

  cudaMemPool_t mempool = nullptr;
  cudaDeviceGetDefaultMemPool(&mempool, local_rank_id);
  uint64_t threshold = UINT64_MAX;
  cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

void EmbeddingManager::SaveSnapshot(const std::string& embedding_name, int64_t local_rank_id,
                                    int64_t rank_id, const std::string& snapshot_name) {
  CudaCurrentDeviceGuard guard(local_rank_id);
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, rank_id);
  std::unique_lock<std::mutex> lock(mutex_);

  auto it = key_value_store_map_.find(map_key);
  CHECK(it != key_value_store_map_.end())
      << "Can not find embedding: " << embedding_name << "-" << rank_id;
  it->second->SaveSnapshot(snapshot_name);
}

void EmbeddingManager::LoadSnapshot(const std::string& embedding_name, int64_t local_rank_id,
                                    int64_t rank_id, const std::string& snapshot_name) {
  CudaCurrentDeviceGuard guard(local_rank_id);
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, rank_id);
  auto it = key_value_store_map_.find(map_key);
  CHECK(it != key_value_store_map_.end())
      << "Can not find embedding: " << embedding_name << "-" << rank_id;
  if (it->second->SnapshotExists(snapshot_name)) {
    it->second->LoadSnapshot(snapshot_name);
  } else {
    LOG(ERROR) << "Here Exists Embedding name is: " << embedding_name << "-" << rank_id
               << " but no corresponding snapshot. ";
  }
}

#endif  // WITH_CUDA

}  // namespace embedding

}  // namespace oneflow
