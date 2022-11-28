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

constexpr size_t kDefaultMaxQueryLength = 131072;

constexpr int64_t kRingBufferSize = 8;

struct IdStatistics {
  IdStatistics() : final_num_unique(0), iter(-1) {}
  uint32_t final_num_unique;
  std::vector<uint32_t> num_unique_matrix;
  int64_t iter;
};

#if CUDA_VERSION >= 11020

class DynamicTmpBufferAllocator final : public TmpBufferAllocator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicTmpBufferAllocator);
  DynamicTmpBufferAllocator(cudaStream_t stream, cudaMemPool_t pool)
      : stream_(stream), mem_pool_(pool) {}
  ~DynamicTmpBufferAllocator() override = default;

  void Allocate(void** ptr, size_t size) override {
    OF_CUDA_CHECK(cudaMallocFromPoolAsync(ptr, GetCudaAlignedSize(size), mem_pool_, stream_));
  }
  void Free(void* ptr) override { OF_CUDA_CHECK(cudaFreeAsync(ptr, stream_)); }

 private:
  cudaStream_t stream_{};
  cudaMemPool_t mem_pool_{};
};

class DynamicAllocationEmbeddingState final : public EmbeddingState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicAllocationEmbeddingState);
  DynamicAllocationEmbeddingState()
      : lookup_values_(nullptr),
        lookup_values_size_(0),
        has_lookup_values_(false),
        lookup_embeddings_(nullptr),
        lookup_embeddings_size_(0),
        has_lookup_embeddings_(false),
        updated_values_(nullptr),
        iter_(-1) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    id_statistics_vec_.resize(kRingBufferSize);
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.handleTypes = cudaMemHandleTypePosixFileDescriptor;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = device_index_;
    cudaMemPoolCreate(&mem_pool_, &poolProps);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &threshold);
  }
  ~DynamicAllocationEmbeddingState() {
    CudaCurrentDeviceGuard guard(device_index_);
    if (has_lookup_values_) { OF_CUDA_CHECK(cudaFree(lookup_values_)); }
    if (has_lookup_embeddings_) { OF_CUDA_CHECK(cudaFree(lookup_embeddings_)); }
    OF_CUDA_CHECK(cudaMemPoolDestroy(mem_pool_));
  }

  std::unique_ptr<TmpBufferAllocator> NewTmpBufferAllocator(
      user_op::KernelComputeContext* ctx) override {
    return std::make_unique<DynamicTmpBufferAllocator>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), mem_pool_);
  }

  void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    iter_ = iter;
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    uint32_t num_unique = this->GetIdNumUnique(iter);
    size_t lookup_values_size =
        GetCudaAlignedSize(num_unique * line_size * GetSizeOfDataType(unique_values->data_type()));
    if (!has_lookup_values_ || lookup_values_size_ < lookup_values_size) {
      if (has_lookup_values_) { OF_CUDA_CHECK(cudaFreeAsync(lookup_values_, cuda_stream)); }
      OF_CUDA_CHECK(
          cudaMallocFromPoolAsync(&lookup_values_, lookup_values_size, mem_pool_, cuda_stream));
      has_lookup_values_ = true;
      lookup_values_size_ = lookup_values_size;
      if (ctx->has_output("embeddings", 0)) {
        user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
        const size_t lookup_embeddings_size = GetCudaAlignedSize(
            num_unique * embedding_size * GetSizeOfDataType(embeddings->data_type()));
        if (!has_lookup_embeddings_ || lookup_embeddings_size_ < lookup_values_size) {
          if (has_lookup_embeddings_) {
            OF_CUDA_CHECK(cudaFreeAsync(lookup_embeddings_, cuda_stream));
          }
          OF_CUDA_CHECK(cudaMallocFromPoolAsync(&lookup_embeddings_, lookup_embeddings_size,
                                                mem_pool_, cuda_stream));
          has_lookup_embeddings_ = true;
          lookup_embeddings_size_ = lookup_embeddings_size;
        }
      } else {
        lookup_embeddings_ = nullptr;
      }
    }
  }

  void* LookupUniqueValues(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_values_);
    return lookup_values_;
  }

  void* LookupEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_embeddings_);
    return lookup_embeddings_;
  }

  void OnEmbeddingLookupEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  void OnEmbeddingGatherStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  const void* EmbeddingGatherIn(int64_t iter) override {
    if (has_lookup_embeddings_) {
      return lookup_embeddings_;
    } else {
      CHECK(has_lookup_values_);
      return lookup_values_;
    }
  }

  void OnEmbeddingGatherEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  void OnEmbeddingShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  const void* EmbeddingShuffleCurRankEmbeddings(int64_t iter) override {
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

  void OnEmbeddingUpdateStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    const int64_t line_size = ctx->Attr<int64_t>("line_size");
    uint32_t num_unique = this->GetIdNumUnique(iter);
    size_t update_values_size = GetCudaAlignedSize(
        num_unique * line_size * GetSizeOfDataType(updated_unique_embeddings->data_type()));
    OF_CUDA_CHECK(cudaMallocFromPoolAsync(&updated_values_, update_values_size, mem_pool_,
                                          ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }

  const void* EmbeddingUpdateUniqueEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_values_);
    return lookup_values_;
  }

  void* EmbeddingUpdateUpdatedUniqueEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return updated_values_;
  }

  void OnEmbeddingUpdateEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  void OnEmbeddingPutStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  const void* EmbeddingPutUniqueEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    return updated_values_;
  }

  void OnEmbeddingPutEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    OF_CUDA_CHECK(
        cudaFreeAsync(updated_values_, ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  }

  void OnEmbeddingFusedUpdatePutStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  const void* EmbeddingFusedUpdatePutUniqueEmbeddings(int64_t iter) override {
    CHECK_EQ(iter_, iter);
    CHECK(has_lookup_values_);
    return lookup_values_;
  }

  void OnEmbeddingFusedUpdatePutEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    // do nothing
  }

  void SetIdFinalNumUnique(uint32_t final_num_unique, int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    id_statistics_vec_.at(index).final_num_unique = final_num_unique;
    id_statistics_vec_.at(index).iter = iter;
  }

  void SetIdNumUniqueMatrix(const std::vector<uint32_t>& num_unique_matrix, int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    id_statistics_vec_.at(index).num_unique_matrix = num_unique_matrix;
    id_statistics_vec_.at(index).iter = iter;
  }

  uint32_t GetIdNumUnique(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const IdStatistics& statistics = id_statistics_vec_.at(index);
    CHECK_EQ(statistics.iter, iter)
        << "saved iter: " << statistics.iter << " current iter: " << iter;
    return statistics.final_num_unique;
  }

  const std::vector<uint32_t>& GetIdNumUniqueMatrix(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const IdStatistics& statistics = id_statistics_vec_.at(index);
    CHECK_EQ(statistics.iter, iter)
        << "saved iter: " << statistics.iter << " current iter: " << iter;
    return statistics.num_unique_matrix;
  }

 private:
  void* lookup_values_;
  size_t lookup_values_size_;
  bool has_lookup_values_;
  void* lookup_embeddings_;
  size_t lookup_embeddings_size_;
  bool has_lookup_embeddings_;
  void* updated_values_;
  int64_t iter_;
  std::vector<IdStatistics> id_statistics_vec_;
  int device_index_{};
  cudaMemPool_t mem_pool_{};
  std::mutex mutex_;
};

#endif

class StaticTmpBufferAllocator final : public TmpBufferAllocator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StaticTmpBufferAllocator);
  StaticTmpBufferAllocator(void* ptr, size_t size) : ptr_(ptr), offset_(0), size_(size) {}
  ~StaticTmpBufferAllocator() override = default;

  void Allocate(void** ptr, size_t size) override {
    CHECK(ptr_ != nullptr);
    CHECK_GE(offset_, 0);
    size_t aligned_size = GetCudaAlignedSize(size);
    CHECK_LE(offset_ + aligned_size, size_);
    *ptr = reinterpret_cast<char*>(ptr_) + offset_;
    offset_ += aligned_size;
  }

  void Free(void* ptr) override {
    // do nothing
  }

 private:
  void* ptr_;
  int64_t offset_;
  size_t size_;
};

class StaticAllocationEmbeddingState final : public EmbeddingState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StaticAllocationEmbeddingState);
  StaticAllocationEmbeddingState()
      : lookup_unique_values_(nullptr),
        lookup_embeddings_(nullptr),
        has_lookup_embeddings_(false),
        embedding_shuffle_cur_rank_embeddings_(nullptr),
        embedding_update_unique_embeddings_(nullptr),
        embedding_update_updated_unique_embeddings_(nullptr),
        embedding_put_unique_embeddings_(nullptr),
        embedding_fused_update_put_unique_embeddings_(nullptr) {
    id_statistics_vec_.resize(kRingBufferSize);
  }
  ~StaticAllocationEmbeddingState() override = default;

  std::unique_ptr<TmpBufferAllocator> NewTmpBufferAllocator(
      user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    return std::make_unique<StaticTmpBufferAllocator>(tmp_buffer->mut_dptr(),
                                                      tmp_buffer->shape_view().elem_cnt());
  }

  void OnEmbeddingLookupStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    user_op::Tensor* unique_values = ctx->Tensor4ArgNameAndIndex("unique_values", 0);
    lookup_unique_values_ = unique_values->mut_dptr();
    if (ctx->has_output("embeddings", 0)) {
      user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
      has_lookup_embeddings_ = true;
      lookup_embeddings_ = embeddings->mut_dptr();
    }
  }

  void* LookupUniqueValues(int64_t iter) override { return lookup_unique_values_; }

  void* LookupEmbeddings(int64_t iter) override {
    CHECK(has_lookup_embeddings_);
    return lookup_embeddings_;
  }

  void OnEmbeddingLookupEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    lookup_unique_values_ = nullptr;
    lookup_embeddings_ = nullptr;
    has_lookup_embeddings_ = false;
  }

  void OnEmbeddingGatherStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    embedding_gather_in_ = in->dptr();
  }

  const void* EmbeddingGatherIn(int64_t iter) override { return embedding_gather_in_; }

  void OnEmbeddingGatherEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    embedding_gather_in_ = nullptr;
  }

  void OnEmbeddingShuffleStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* cur_rank_embeddings =
        ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0);
    embedding_shuffle_cur_rank_embeddings_ = cur_rank_embeddings->dptr();
  }

  const void* EmbeddingShuffleCurRankEmbeddings(int64_t iter) override {
    return embedding_shuffle_cur_rank_embeddings_;
  }

  void OnEmbeddingShuffleEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    embedding_shuffle_cur_rank_embeddings_ = nullptr;
  }

  void OnEmbeddingUpdateStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    embedding_update_unique_embeddings_ = unique_embeddings->dptr();
    embedding_update_updated_unique_embeddings_ = updated_unique_embeddings->mut_dptr();
  }

  const void* EmbeddingUpdateUniqueEmbeddings(int64_t iter) override {
    return embedding_update_unique_embeddings_;
  }

  void* EmbeddingUpdateUpdatedUniqueEmbeddings(int64_t iter) override {
    return embedding_update_updated_unique_embeddings_;
  }

  void OnEmbeddingUpdateEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    embedding_update_unique_embeddings_ = nullptr;
    embedding_update_updated_unique_embeddings_ = nullptr;
  }

  void OnEmbeddingPutStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    embedding_put_unique_embeddings_ = unique_embeddings->dptr();
  }

  const void* EmbeddingPutUniqueEmbeddings(int64_t iter) override {
    return embedding_put_unique_embeddings_;
  }

  void OnEmbeddingPutEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    embedding_put_unique_embeddings_ = nullptr;
  }

  void OnEmbeddingFusedUpdatePutStart(user_op::KernelComputeContext* ctx, int64_t iter) override {
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    embedding_fused_update_put_unique_embeddings_ = unique_embeddings->dptr();
  }

  const void* EmbeddingFusedUpdatePutUniqueEmbeddings(int64_t iter) override {
    return embedding_fused_update_put_unique_embeddings_;
  }

  void OnEmbeddingFusedUpdatePutEnd(user_op::KernelComputeContext* ctx, int64_t iter) override {
    embedding_fused_update_put_unique_embeddings_ = nullptr;
  }

  void SetIdFinalNumUnique(uint32_t final_num_unique, int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    id_statistics_vec_.at(index).final_num_unique = final_num_unique;
    id_statistics_vec_.at(index).iter = iter;
  }

  void SetIdNumUniqueMatrix(const std::vector<uint32_t>& num_unique_matrix, int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    id_statistics_vec_.at(index).num_unique_matrix = num_unique_matrix;
    id_statistics_vec_.at(index).iter = iter;
  }

  uint32_t GetIdNumUnique(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const IdStatistics& statistics = id_statistics_vec_.at(index);
    CHECK_EQ(statistics.iter, iter)
        << "saved iter: " << statistics.iter << " current iter: " << iter;
    return statistics.final_num_unique;
  }

  const std::vector<uint32_t>& GetIdNumUniqueMatrix(int64_t iter) override {
    std::unique_lock<std::mutex> lock(mutex_);
    int64_t index = iter % kRingBufferSize;
    const IdStatistics& statistics = id_statistics_vec_.at(index);
    CHECK_EQ(statistics.iter, iter)
        << "saved iter: " << statistics.iter << " current iter: " << iter;
    return statistics.num_unique_matrix;
  }

  void* lookup_unique_values_;
  void* lookup_embeddings_;
  bool has_lookup_embeddings_;
  const void* embedding_gather_in_;
  const void* embedding_shuffle_cur_rank_embeddings_;
  const void* embedding_update_unique_embeddings_;
  void* embedding_update_updated_unique_embeddings_;
  const void* embedding_put_unique_embeddings_;
  const void* embedding_fused_update_put_unique_embeddings_;
  std::vector<IdStatistics> id_statistics_vec_;
  std::mutex mutex_;
};

EmbeddingState* EmbeddingManager::GetEmbeddingState(const std::string& embedding_name,
                                                    int64_t rank_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, rank_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = embedding_state_map_.find(map_key);
  // for id shuffle test, not need to create table
  if (it == embedding_state_map_.end()) {
    LOG(INFO) << "create embedding state: " << embedding_name << "-" << rank_id;
    if (UseDynamicMemoryAllocation()) {
#if CUDA_VERSION >= 11020
      it =
          embedding_state_map_.emplace(map_key, std::make_unique<DynamicAllocationEmbeddingState>())
              .first;
#else
      UNIMPLEMENTED();
#endif
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

  if (UseDynamicMemoryAllocation()) {
#if CUDA_VERSION >= 11020
    CHECK(embedding_state_map_.emplace(map_key, std::make_unique<DynamicAllocationEmbeddingState>())
              .second)
        << "Can't create an embedding state with same name of an existing embedding, the name: "
        << name;
#else
    UNIMPLEMENTED();
#endif
  } else {
    CHECK(embedding_state_map_.emplace(map_key, std::make_unique<StaticAllocationEmbeddingState>())
              .second)
        << "Can't create an embedding state with same name of an existing embedding, the name: "
        << name;
  }
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
