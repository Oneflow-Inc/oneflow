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
#include "oneflow/core/embedding/persistent_table_key_value_store.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/embedding/persistent_table.h"
#include <robin_hood.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>

namespace oneflow {

namespace embedding {

namespace {

class IteratorImpl : public KVIterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IteratorImpl);
  IteratorImpl(PersistentTable::Iterator* base_iter, uint32_t key_size, uint32_t value_size,
               uint32_t max_query_length, void* host_keys_buffer, void* host_values_buffer,
               uint32_t* host_num_buffer)
      : base_iter_(base_iter),
        key_size_(key_size),
        value_size_(value_size),
        max_query_length_(max_query_length),
        host_keys_buffer_(host_keys_buffer),
        host_values_buffer_(host_values_buffer),
        host_num_buffer_(host_num_buffer) {}
  ~IteratorImpl() override = default;

  void NextN(ep::Stream* stream, uint32_t n_request, uint32_t* n_result, void* keys,
             void* values) override {
    CHECK_LE(n_request, max_query_length_);
    auto cuda_stream = stream->As<ep::CudaStream>();
    CHECK_JUST(cuda_stream->Sync());
    base_iter_->Next(n_request, host_num_buffer_, host_keys_buffer_, host_values_buffer_);
    OF_CUDA_CHECK(cudaMemcpyAsync(n_result, host_num_buffer_, sizeof(uint32_t), cudaMemcpyDefault,
                                  cuda_stream->cuda_stream()));
    const uint32_t num_keys = *host_num_buffer_;
    if (num_keys != 0) {
      OF_CUDA_CHECK(cudaMemcpyAsync(keys, host_keys_buffer_, num_keys * key_size_,
                                    cudaMemcpyDefault, cuda_stream->cuda_stream()));
      OF_CUDA_CHECK(cudaMemcpyAsync(values, host_values_buffer_, num_keys * value_size_,
                                    cudaMemcpyDefault, cuda_stream->cuda_stream()));
    }
  }

  void Reset() override { base_iter_->Reset(); }

 private:
  PersistentTable::Iterator* base_iter_;
  uint32_t key_size_;
  uint32_t value_size_;
  uint32_t max_query_length_;
  void* host_keys_buffer_;
  void* host_values_buffer_;
  uint32_t* host_num_buffer_;
};

template<typename Key>
class KeyValueStoreImpl : public KeyValueStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyValueStoreImpl);
  explicit KeyValueStoreImpl(const PersistentTableKeyValueStoreOptions& options)
      : device_index_(-1), max_query_length_(0) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    key_size_ = options.table_options.key_size;
    value_size_ = options.table_options.value_size;
    table_ = NewPersistentTable(options.table_options);
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(
        device_index_, reinterpret_cast<void**>(&host_query_keys_), key_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_,
                                          reinterpret_cast<void**>(&host_query_values_),
                                          value_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_, reinterpret_cast<void**>(&host_n_missing_),
                                          sizeof(uint32_t)));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_,
                                          reinterpret_cast<void**>(&host_missing_indices_),
                                          sizeof(uint32_t) * max_query_length_));
  }
  ~KeyValueStoreImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    if (max_query_length_ != 0) {
      OF_CUDA_CHECK(cudaFreeHost(host_query_keys_));
      OF_CUDA_CHECK(cudaFreeHost(host_query_values_));
      OF_CUDA_CHECK(cudaFreeHost(host_missing_indices_));
    }
    OF_CUDA_CHECK(cudaFreeHost(host_n_missing_));
  }

  uint32_t KeySize() const override { return key_size_; }

  uint32_t ValueSize() const override { return value_size_; }

  uint32_t MaxQueryLength() const override { return max_query_length_; }

  void ReserveQueryLength(uint32_t query_length) override {
    CudaCurrentDeviceGuard guard(device_index_);
    if (query_length <= max_query_length_) { return; }
    if (max_query_length_ != 0) {
      OF_CUDA_CHECK(cudaFreeHost(host_query_keys_));
      OF_CUDA_CHECK(cudaFreeHost(host_query_values_));
      OF_CUDA_CHECK(cudaFreeHost(host_missing_indices_));
    }
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(
        device_index_, reinterpret_cast<void**>(&host_query_keys_), key_size_ * query_length));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(
        device_index_, reinterpret_cast<void**>(&host_query_values_), value_size_ * query_length));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_,
                                          reinterpret_cast<void**>(&host_missing_indices_),
                                          sizeof(uint32_t) * query_length));
    max_query_length_ = query_length;
  }

  using KeyValueStore::Get;
  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint32_t* n_missing, uint32_t* missing_indices) override;
  void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values) override;
  bool SnapshotExists(const std::string& name) override;
  void LoadSnapshot(const std::string& name) override;
  void LoadSnapshot(const std::string& name,
                    const std::function<void(KVIterator* iter)>& Hook) override;
  void SaveSnapshot(const std::string& name) override;

 private:
  int device_index_;
  uint32_t max_query_length_;
  uint32_t key_size_;
  uint32_t value_size_;
  Key* host_query_keys_{};
  uint8_t* host_query_values_{};
  uint32_t* host_n_missing_{};
  uint32_t* host_missing_indices_{};

  std::mutex mutex_;
  std::unique_ptr<PersistentTable> table_;
};

template<typename Key>
void KeyValueStoreImpl<Key>::Get(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                 void* values, uint32_t* n_missing, uint32_t* missing_indices) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto cuda_stream = stream->As<ep::CudaStream>();
  CHECK_LE(num_keys, max_query_length_);
  if (num_keys == 0) {
    OF_CUDA_CHECK(cudaMemsetAsync(n_missing, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
    return;
  }
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_keys_, keys, key_size_ * num_keys, cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());

  table_->Get(num_keys, host_query_keys_, host_query_values_, host_n_missing_,
              host_missing_indices_);

  OF_CUDA_CHECK(cudaMemcpyAsync(values, host_query_values_, num_keys * value_size_,
                                cudaMemcpyDefault, cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(n_missing, host_n_missing_, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(missing_indices, host_missing_indices_,
                                (*host_n_missing_) * sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
}

template<typename Key>
void KeyValueStoreImpl<Key>::Put(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                 const void* values) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto cuda_stream = stream->As<ep::CudaStream>();
  CHECK_LE(num_keys, max_query_length_);
  if (num_keys == 0) { return; }
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_keys_, keys, key_size_ * num_keys, cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_values_, values, value_size_ * num_keys,
                                cudaMemcpyDefault, cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  table_->Put(num_keys, host_query_keys_, host_query_values_);
}

template<typename Key>
bool KeyValueStoreImpl<Key>::SnapshotExists(const std::string& name) {
  return table_->SnapshotExists(name);
}

template<typename Key>
void KeyValueStoreImpl<Key>::LoadSnapshot(const std::string& name) {
  CudaCurrentDeviceGuard guard(device_index_);
  LoadSnapshot(name, nullptr);
}

template<typename Key>
void KeyValueStoreImpl<Key>::LoadSnapshot(const std::string& name,
                                          const std::function<void(KVIterator* iter)>& Hook) {
  CudaCurrentDeviceGuard guard(device_index_);
  if (Hook) {
    table_->LoadSnapshot(name, [&](PersistentTable::Iterator* chunk_iterator) {
      IteratorImpl iterator(chunk_iterator, KeySize(), ValueSize(), max_query_length_,
                            host_query_keys_, host_query_values_, host_n_missing_);
      Hook(&iterator);
    });
  } else {
    table_->LoadSnapshot(name);
  }
}

template<typename Key>
void KeyValueStoreImpl<Key>::SaveSnapshot(const std::string& name) {
  CudaCurrentDeviceGuard guard(device_index_);
  table_->SaveSnapshot(name);
}

}  // namespace

std::unique_ptr<KeyValueStore> NewPersistentTableKeyValueStore(
    const PersistentTableKeyValueStoreOptions& options) {
  if (options.table_options.key_size == sizeof(uint64_t)) {
    return std::unique_ptr<KeyValueStore>(new KeyValueStoreImpl<uint64_t>(options));
  } else if (options.table_options.key_size == sizeof(uint32_t)) {
    return std::unique_ptr<KeyValueStore>(new KeyValueStoreImpl<uint32_t>(options));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace embedding

}  // namespace oneflow
