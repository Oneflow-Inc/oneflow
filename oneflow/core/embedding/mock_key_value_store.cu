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
#include "oneflow/core/embedding/mock_key_value_store.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace embedding {

namespace {

template<typename Key>
class IteratorImpl : public KVIterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IteratorImpl);
  IteratorImpl(HashMap<Key, std::string>* store, uint32_t key_size, uint32_t value_size,
               uint32_t max_query_length, void* host_keys_buffer, void* host_values_buffer,
               uint32_t* host_num_buffer)
      : store_(store),
        pos_(store->begin()),
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
    *host_num_buffer_ = 0;
    while (*host_num_buffer_ < n_request && pos_ != store_->end()) {
      reinterpret_cast<Key*>(host_keys_buffer_)[*host_num_buffer_] = pos_->first;
      std::memcpy(reinterpret_cast<char*>(host_values_buffer_) + *host_num_buffer_ * value_size_,
                  pos_->second.data(), value_size_);
    }
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

  void Reset() override { pos_ = store_->begin(); }

 private:
  HashMap<Key, std::string>* store_;
  typename HashMap<Key, std::string>::iterator pos_;
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
  explicit KeyValueStoreImpl(const MockKeyValueStoreOptions& options)
      : device_index_(-1), max_query_length_(0) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    key_size_ = options.key_size;
    value_size_ = options.value_size;
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
  HashMap<Key, std::string> store_;
  HashMap<std::string, HashMap<Key, std::string>> snapshots_;
  std::mutex mutex_;
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
  *host_n_missing_ = 0;
  for (uint32_t i = 0; i < num_keys; ++i) {
    auto it = store_.find(host_query_keys_[i]);
    if (it != store_.end()) {
      std::memcpy(host_query_values_ + i * value_size_, it->second.data(), value_size_);
    } else {
      host_missing_indices_[*host_n_missing_] = i;
      *host_n_missing_ += 1;
    }
  }
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
  for (uint32_t i = 0; i < num_keys; ++i) {
    store_[host_query_keys_[i]] = std::string(
        reinterpret_cast<const char*>(host_query_values_) + i * value_size_, value_size_);
  }
}

template<typename Key>
bool KeyValueStoreImpl<Key>::SnapshotExists(const std::string& name) {
  return snapshots_.find(name) != snapshots_.end();
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
  store_ = snapshots_[name];
  if (Hook) {
    IteratorImpl<Key> iterator(&store_, KeySize(), ValueSize(), max_query_length_, host_query_keys_,
                               host_query_values_, host_n_missing_);
    Hook(&iterator);
  }
}

template<typename Key>
void KeyValueStoreImpl<Key>::SaveSnapshot(const std::string& name) {
  CudaCurrentDeviceGuard guard(device_index_);
  snapshots_[name] = store_;
}

}  // namespace

std::unique_ptr<KeyValueStore> NewMockKeyValueStore(const MockKeyValueStoreOptions& options) {
  if (options.key_size == sizeof(uint64_t)) {
    return std::unique_ptr<KeyValueStore>(new KeyValueStoreImpl<uint64_t>(options));
  } else if (options.key_size == sizeof(uint32_t)) {
    return std::unique_ptr<KeyValueStore>(new KeyValueStoreImpl<uint32_t>(options));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace embedding

}  // namespace oneflow
