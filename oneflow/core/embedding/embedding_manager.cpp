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

ValuesPtr* EmbeddingManager::GetValuesPtr(const std::string& embedding_name, int64_t rank_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, rank_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = values_ptrs_map_.find(map_key);
  CHECK(it != values_ptrs_map_.end())
      << "Can not find embedding: " << embedding_name << "-" << rank_id;
  return it->second.get();
}

NumUniques* EmbeddingManager::GetNumUniques(const std::string& embedding_name, int64_t rank_id) {
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, rank_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = num_uniques_map_.find(map_key);
  // for id shuffle test, not need to create table
  if (it == num_uniques_map_.end()) {
    it = num_uniques_map_.emplace(map_key, std::make_unique<NumUniques>()).first;
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

  CHECK(num_uniques_map_.emplace(map_key, std::make_unique<NumUniques>()).second)
      << "Can't create an embedding values with same name of an existing embedding, the name: "
      << name;
  if (UseDynamicMemoryAllocation()) {
#if CUDA_VERSION >= 11020
    CHECK(values_ptrs_map_.emplace(map_key, std::make_unique<ValuesPtr>()).second)
        << "Can't create an embedding values with same name of an existing embedding, the name: "
        << name;

    cudaMemPool_t mempool = nullptr;
    cudaDeviceGetDefaultMemPool(&mempool, local_rank_id);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
#else
    UNIMPLEMENTED();
#endif
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

uint32_t NumUniques::GetNumUnique(int64_t iter) {
  std::unique_lock<std::mutex> lock(mutex_);
  int64_t index = iter % kRingBufferSize;
  CHECK_EQ(num_unique_iter_.at(index), iter)
      << "saved iter: " << num_unique_iter_.at(index) << " current iter: " << iter;
  return num_unique_.at(index);
}

void NumUniques::SetNumUnique(uint32_t num_unique, int64_t iter) {
  std::unique_lock<std::mutex> lock(mutex_);
  int64_t index = iter % kRingBufferSize;
  num_unique_.at(index) = num_unique;
  num_unique_iter_.at(index) = iter;
}

void NumUniques::SetNumUniqueMatrix(const std::vector<uint32_t>& num_unique_matrix, int64_t iter) {
  std::unique_lock<std::mutex> lock(mutex_);
  int64_t index = iter % kRingBufferSize;
  num_unique_matrix_.at(index) = num_unique_matrix;
  num_unique_matrix_iter_.at(index) = iter;
}

const std::vector<uint32_t>& NumUniques::GetNumUniqueMatrix(int64_t iter) {
  std::unique_lock<std::mutex> lock(mutex_);
  int64_t index = iter % kRingBufferSize;
  CHECK_EQ(num_unique_matrix_iter_.at(index), iter)
      << "saved iter: " << num_unique_matrix_iter_.at(index) << " current iter: " << iter;
  return num_unique_matrix_.at(index);
}

void* ValuesPtr::GetLookupValuesPtr(int64_t iter) {
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(has_lookup_values_);
  CHECK_EQ(lookup_values_iter_, iter)
      << "saved iter: " << lookup_values_iter_ << " current iter: " << iter;
  return lookup_values_;
}
void* ValuesPtr::GetLookupEmbeddingsPtr(int64_t iter) {
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(has_lookup_embeddings_);
  CHECK_EQ(lookup_embeddings_iter_, iter)
      << "saved iter: " << lookup_embeddings_iter_ << " current iter: " << iter;
  return lookup_embeddings_;
}
void* ValuesPtr::MallocLookupValuesPtr(int64_t iter, size_t data_size, cudaStream_t cuda_stream) {
#if CUDA_VERSION >= 11020
  std::unique_lock<std::mutex> lock(mutex_);
  if (has_lookup_values_ && lookup_values_size_ >= data_size) {
    // do nothing
  } else {
    if (has_lookup_values_) { OF_CUDA_CHECK(cudaFreeAsync(lookup_values_, cuda_stream)); }
    OF_CUDA_CHECK(cudaMallocAsync(&(lookup_values_), data_size, cuda_stream));
    has_lookup_values_ = true;
    lookup_values_size_ = data_size;
  }
  lookup_values_iter_ = iter;
  return lookup_values_;
#else
  UNIMPLEMENTED();
  return nullptr;
#endif
}
bool ValuesPtr::HasLookupEmbeddings() { return has_lookup_embeddings_; }

void* ValuesPtr::MallocLookupEmbeddingsPtr(int64_t iter, size_t data_size,
                                           cudaStream_t cuda_stream) {
#if CUDA_VERSION >= 11020
  std::unique_lock<std::mutex> lock(mutex_);
  if (has_lookup_embeddings_ && lookup_embeddings_size_ >= data_size) {
    // do nothing
  } else {
    if (has_lookup_embeddings_) { OF_CUDA_CHECK(cudaFreeAsync(lookup_embeddings_, cuda_stream)); }
    OF_CUDA_CHECK(cudaMallocAsync(&lookup_embeddings_, data_size, cuda_stream));
    has_lookup_embeddings_ = true;
    lookup_embeddings_size_ = data_size;
  }
  lookup_embeddings_iter_ = iter;
  return lookup_embeddings_;
#else
  UNIMPLEMENTED();
  return nullptr;
#endif
}

void* ValuesPtr::MallocUpdatedValuesPtr(int64_t iter, size_t data_size, cudaStream_t cuda_stream) {
#if CUDA_VERSION >= 11020
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(!has_updated_values_);
  OF_CUDA_CHECK(cudaMallocAsync(&updated_values_, data_size, cuda_stream));
  has_updated_values_ = true;
  updated_values_iter_ = iter;
  return updated_values_;
#else
  UNIMPLEMENTED();
  return nullptr;
#endif
}
void* ValuesPtr::GetUpdatedValuesPtr(int64_t iter) {
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(has_updated_values_);
  CHECK_EQ(updated_values_iter_, iter)
      << "saved iter: " << updated_values_iter_ << " current iter: " << iter;
  return updated_values_;
}
void ValuesPtr::FreeUpdatedValuesPtr(int64_t iter, cudaStream_t cuda_stream) {
#if CUDA_VERSION >= 11020
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(has_updated_values_);
  CHECK_EQ(updated_values_iter_, iter)
      << "saved iter: " << updated_values_iter_ << " current iter: " << iter;
  OF_CUDA_CHECK(cudaFreeAsync(updated_values_, cuda_stream));
  has_updated_values_ = false;
#else
  UNIMPLEMENTED();
#endif
}

#endif  // WITH_CUDA

}  // namespace embedding

}  // namespace oneflow
