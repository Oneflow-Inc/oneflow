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
#include "oneflow/core/embedding/cached_key_value_store.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace embedding {

namespace {
template<typename Key, typename Elem>
__global__ void PostStoreGetKernel(uint32_t num_cache_missing, uint32_t num_store_missing,
                                   uint32_t num_elems_per_value,
                                   const uint32_t* cache_missing_indices,
                                   const uint32_t* store_missing_indices, const Elem* store_values,
                                   Elem* values, uint32_t* missing_indices) {
  const uint32_t num_cache_missing_elem = num_cache_missing * num_elems_per_value;
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_cache_missing_elem) {
    const uint32_t value_index = i / num_elems_per_value;
    const uint32_t elem_index = i - value_index * num_elems_per_value;
    values[cache_missing_indices[value_index] * num_elems_per_value + elem_index] = store_values[i];
  }
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_store_missing) {
    missing_indices[i] = cache_missing_indices[store_missing_indices[i]];
  }
}

template<typename Key, typename Elem>
class CacheKeyValueStoreImpl : public KeyValueStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CacheKeyValueStoreImpl);
  CacheKeyValueStoreImpl(std::unique_ptr<KeyValueStore>&& store, std::unique_ptr<Cache>&& cache)
      : store_(std::move(store)), cache_(std::move(cache)), synced_(true), max_query_length_(0) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    CHECK_EQ(store_->KeySize(), cache_->KeySize());
    CHECK_EQ(store_->ValueSize(), cache_->ValueSize());
    OF_CUDA_CHECK(cudaMalloc(&num_buffer_, sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMallocHost(&host_num_buffer_, sizeof(uint32_t)));
    num_elems_per_value_ = store_->ValueSize() / sizeof(Elem);
  }
  ~CacheKeyValueStoreImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(num_buffer_));
    OF_CUDA_CHECK(cudaFreeHost(host_num_buffer_));
    if (max_query_length_ != 0) {
      OF_CUDA_CHECK(cudaFree(keys_buffer_));
      OF_CUDA_CHECK(cudaFree(values_buffer_));
      OF_CUDA_CHECK(cudaFree(indices_buffer0_));
      OF_CUDA_CHECK(cudaFree(indices_buffer1_));
    }
    cache_.reset();
    store_.reset();
  }

  uint32_t KeySize() const override { return store_->KeySize(); }
  uint32_t ValueSize() const override { return store_->ValueSize(); }
  uint32_t MaxQueryLength() const override { return max_query_length_; }

  void ReserveQueryLength(uint32_t query_length) override {
    CudaCurrentDeviceGuard guard(device_index_);
    if (query_length <= max_query_length_) { return; }
    if (query_length > cache_->MaxQueryLength()) { cache_->ReserveQueryLength(query_length); }
    if (query_length > store_->MaxQueryLength()) { store_->ReserveQueryLength(query_length); }
    if (max_query_length_ != 0) {
      OF_CUDA_CHECK(cudaFree(keys_buffer_));
      OF_CUDA_CHECK(cudaFree(values_buffer_));
      OF_CUDA_CHECK(cudaFree(indices_buffer0_));
      OF_CUDA_CHECK(cudaFree(indices_buffer1_));
    }
    OF_CUDA_CHECK(cudaMalloc(&keys_buffer_, query_length * store_->KeySize()));
    OF_CUDA_CHECK(cudaMalloc(&values_buffer_, query_length * store_->ValueSize()));
    OF_CUDA_CHECK(cudaMalloc(&indices_buffer0_, query_length * sizeof(uint32_t)));
    OF_CUDA_CHECK(cudaMalloc(&indices_buffer1_, query_length * sizeof(uint32_t)));
    max_query_length_ = query_length;
  }

  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint32_t* n_missing, uint32_t* missing_indices) override;
  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint8_t* mask) override;
  void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values) override;
  void FusedHalfUpdatePut(ep::Stream* stream, uint32_t n_keys, const void* keys, const void* values,
                          const void* update, const float* lr, float scale) override;
  bool IsFusionSupported() override {
    return cache_->Policy() == CacheOptions::Policy::kFull
           && cache_->ValueType() == DataType::kFloat;
  }
  bool SnapshotExists(const std::string& name) override;
  void LoadSnapshot(const std::string& name) override;
  void SaveSnapshot(const std::string& name) override;
  void LoadSnapshot(const std::string& name,
                    const std::function<void(KVIterator* iter)>& Hook) override;

 private:
  void SyncCacheToStore();

  std::unique_ptr<KeyValueStore> store_;
  std::unique_ptr<Cache> cache_;

  uint32_t* num_buffer_{};
  uint32_t* host_num_buffer_{};
  Key* keys_buffer_{};
  Elem* values_buffer_{};
  uint32_t* indices_buffer0_{};
  uint32_t* indices_buffer1_{};
  int device_index_{};
  uint32_t max_query_length_;
  uint32_t num_elems_per_value_{};
  std::recursive_mutex mutex_;
  bool synced_;
};

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::Get(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                            void* values, uint32_t* n_missing,
                                            uint32_t* missing_indices) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto cuda_stream = stream->As<ep::CudaStream>();
  if (cache_->Policy() == CacheOptions::Policy::kFull) {
    cache_->Get(stream, num_keys, keys, values, n_missing, keys_buffer_, missing_indices);
    return;
  } else {
    cache_->Get(stream, num_keys, keys, values, num_buffer_, keys_buffer_, indices_buffer0_);
  }
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, num_buffer_, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  const uint32_t num_cache_missing = *host_num_buffer_;
  if (num_cache_missing == 0) {
    OF_CUDA_CHECK(cudaMemsetAsync(n_missing, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
    return;
  }
  store_->Get(stream, num_cache_missing, keys_buffer_, values_buffer_, n_missing, indices_buffer1_);
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, n_missing, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  const uint32_t num_store_missing = *host_num_buffer_;
  RUN_CUDA_KERNEL((PostStoreGetKernel<Key, Elem>), stream, num_cache_missing * num_elems_per_value_,
                  num_cache_missing, num_store_missing, num_elems_per_value_, indices_buffer0_,
                  indices_buffer1_, values_buffer_, static_cast<Elem*>(values), missing_indices);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::Get(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                            void* values, uint8_t* mask) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (cache_->Policy() == CacheOptions::Policy::kFull) {
    cache_->Get(stream, num_keys, keys, values, mask);
    return;
  } else {
    UNIMPLEMENTED();
  }
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::Put(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                            const void* values) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  synced_ = false;
  auto cuda_stream = stream->As<ep::CudaStream>();
  if (cache_->Policy() != CacheOptions::Policy::kFull) {
    OF_CUDA_CHECK(cudaMemsetAsync(num_buffer_, 0, sizeof(uint32_t), cuda_stream->cuda_stream()));
  }
  cache_->Put(stream, num_keys, keys, values, num_buffer_, keys_buffer_, values_buffer_);
  if (cache_->Policy() == CacheOptions::Policy::kFull) { return; }
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, num_buffer_, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  store_->Put(stream, *host_num_buffer_, keys_buffer_, values_buffer_);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::FusedHalfUpdatePut(ep::Stream* stream, uint32_t num_keys,
                                                           const void* keys, const void* values,
                                                           const void* update, const float* lr,
                                                           float scale) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (cache_->Policy() != CacheOptions::Policy::kFull) {
    OF_CUDA_CHECK(cudaMemsetAsync(num_buffer_, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
  }
  if (cache_->Policy() != CacheOptions::Policy::kFull || cache_->ValueType() != DataType::kFloat) {
    UNIMPLEMENTED();
  }
  synced_ = false;
  cache_->FusedHalfUpdatePut(stream, num_keys, keys, values, update, lr, scale, num_buffer_,
                             keys_buffer_, values_buffer_);
}

template<typename Key, typename Elem>
bool CacheKeyValueStoreImpl<Key, Elem>::SnapshotExists(const std::string& name) {
  return store_->SnapshotExists(name);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::LoadSnapshot(const std::string& name) {
  LoadSnapshot(name, nullptr);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::LoadSnapshot(
    const std::string& name, const std::function<void(KVIterator* iter)>& Hook) {
  CudaCurrentDeviceGuard guard(device_index_);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  CHECK_GT(max_query_length_, 0);
  cache_->Clear();
  auto device =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA, device_index_);
  CHECK(device);
  auto* stream = device->CreateStream();
  store_->LoadSnapshot(name, [&](KVIterator* iter) {
    if (cache_->Policy() == CacheOptions::Policy::kFull) {
      auto* cuda_stream = stream->As<ep::CudaStream>();
      while (true) {
        iter->NextN(stream, max_query_length_, num_buffer_, keys_buffer_, values_buffer_);
        OF_CUDA_CHECK(cudaDeviceSynchronize());
        OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, num_buffer_, sizeof(uint32_t),
                                      cudaMemcpyDefault, cuda_stream->cuda_stream()));
        CHECK_JUST(stream->Sync());
        if (*host_num_buffer_ == 0) { return; }
        cache_->Put(stream, *host_num_buffer_, keys_buffer_, values_buffer_, num_buffer_, nullptr,
                    nullptr);
      }
    }
    if (Hook) {
      iter->Reset();
      Hook(iter);
    }
  });
  device->DestroyStream(stream);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::SaveSnapshot(const std::string& name) {
  CudaCurrentDeviceGuard guard(device_index_);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  SyncCacheToStore();
  store_->SaveSnapshot(name);
}

template<typename Key, typename Elem>
void CacheKeyValueStoreImpl<Key, Elem>::SyncCacheToStore() {
  if (synced_) { return; }
  CudaCurrentDeviceGuard guard(device_index_);
  auto device =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA, device_index_);
  CHECK(device);
  auto* stream = device->CreateStream();
  auto* cuda_stream = stream->As<ep::CudaStream>();
  const uint64_t dump_capacity = cache_->DumpCapacity();
  CHECK_GT(max_query_length_, 0);
  for (uint64_t start_key_index = 0; start_key_index < dump_capacity;
       start_key_index += max_query_length_) {
    cache_->Dump(stream, start_key_index,
                 std::min(start_key_index + max_query_length_, dump_capacity), num_buffer_,
                 keys_buffer_, values_buffer_);
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_buffer_, num_buffer_, sizeof(uint32_t),
                                  cudaMemcpyDefault, cuda_stream->cuda_stream()));
    CHECK_JUST(stream->Sync());
    if (*host_num_buffer_ == 0) { continue; }
    store_->Put(stream, *host_num_buffer_, keys_buffer_, values_buffer_);
    CHECK_JUST(stream->Sync());
  }
  cache_->ClearDirtyFlags();
  device->DestroyStream(stream);
  synced_ = true;
}

template<typename Key>
std::unique_ptr<KeyValueStore> DispatchElemType(std::unique_ptr<KeyValueStore>&& store,
                                                std::unique_ptr<Cache>&& cache) {
  const uint32_t value_size = store->ValueSize();
  if (value_size % sizeof(uint4) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint4>(std::move(store), std::move(cache)));
  } else if (value_size % sizeof(uint64_t) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint64_t>(std::move(store), std::move(cache)));
  } else if (value_size % sizeof(uint32_t) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint32_t>(std::move(store), std::move(cache)));
  } else if (value_size % sizeof(uint16_t) == 0) {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint16_t>(std::move(store), std::move(cache)));
  } else {
    return std::unique_ptr<KeyValueStore>(
        new CacheKeyValueStoreImpl<Key, uint8_t>(std::move(store), std::move(cache)));
  }
}

std::unique_ptr<KeyValueStore> DispatchKeyType(std::unique_ptr<KeyValueStore>&& store,
                                               std::unique_ptr<Cache>&& cache) {
  const uint32_t key_size = store->KeySize();
  if (key_size == 4) {
    return DispatchElemType<uint32_t>(std::move(store), std::move(cache));
  } else if (key_size == 8) {
    return DispatchElemType<uint64_t>(std::move(store), std::move(cache));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace

std::unique_ptr<KeyValueStore> NewCachedKeyValueStore(std::unique_ptr<KeyValueStore>&& store,
                                                      std::unique_ptr<Cache>&& cache) {
  return DispatchKeyType(std::move(store), std::move(cache));
}

}  // namespace embedding

}  // namespace oneflow
