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
#include "oneflow/core/embedding/cuda_in_memory_key_value_store.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace embedding {

namespace {

template<typename Key>
__global__ void PrefetchPlainEncodingKernel(Key num_shards, uint32_t num_keys, const Key* keys,
                                            uint64_t* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) { context[i] = keys[i] / num_shards; }
}

template<typename Elem>
__global__ void LookupKernel(uint32_t vec_size, uint64_t num_embeddings,
                             uint64_t num_device_embeddings, const Elem* device_embeddings,
                             const Elem* host_embeddings, uint32_t values_elem_cnt,
                             const uint64_t* context, Elem* values) {
  CUDA_1D_KERNEL_LOOP(i, values_elem_cnt) {
    const uint64_t key_id = i / vec_size;
    const uint64_t row_id = context[key_id];
    const uint64_t col_id = i - key_id * vec_size;
    Elem elem;
    if (row_id < num_device_embeddings) {
      elem = device_embeddings[row_id * vec_size + col_id];
    } else if (row_id < num_embeddings) {
      elem = host_embeddings[(row_id - num_device_embeddings) * vec_size + col_id];
    } else {
      elem = 0;
    }
    values[i] = elem;
  }
}

template<typename Elem>
__global__ void UpdateKernel(uint32_t vec_size, uint64_t num_embeddings,
                             uint64_t num_device_embeddings, Elem* device_embeddings,
                             Elem* host_embeddings, uint32_t values_elem_cnt,
                             const uint64_t* context, const Elem* values) {
  CUDA_1D_KERNEL_LOOP(i, values_elem_cnt) {
    const uint64_t key_id = i / vec_size;
    const uint64_t row_id = context[key_id];
    const uint64_t col_id = i - key_id * vec_size;
    const Elem elem = values[i];
    if (row_id < num_device_embeddings) {
      device_embeddings[row_id * vec_size + col_id] = elem;
    } else if (row_id < num_embeddings) {
      host_embeddings[(row_id - num_device_embeddings) * vec_size + col_id] = elem;
    } else {
      // do nothing;
    }
  }
}

template<typename Key>
class PlainEncoder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PlainEncoder);
  explicit PlainEncoder(const CudaInMemoryKeyValueStoreOptions& options)
      : num_shards_(options.num_shards) {}
  ~PlainEncoder() = default;

  void Encode(ep::Stream* stream, uint32_t num_keys, const void* keys, uint64_t* context) {
    RUN_CUDA_KERNEL((PrefetchPlainEncodingKernel<Key>), stream, num_keys, num_shards_, num_keys,
                    static_cast<const Key*>(keys), context);
  }

  size_t WorkspaceSize() const { return 0; }

 private:
  uint32_t num_shards_;
};

template<typename Key>
class OrdinalEncoder {
  OF_DISALLOW_COPY_AND_MOVE(OrdinalEncoder);
  OrdinalEncoder() = default;
  ~OrdinalEncoder() = default;

  void Encode(ep::Stream* stream, uint32_t num_keys, const void* keys, uint64_t* context) {}

  size_t WorkspaceSize() const { return 0; }
};

template<typename Encoder, typename Key, typename Elem>
class KeyValueStoreImpl : public KeyValueStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyValueStoreImpl);
  explicit KeyValueStoreImpl(const CudaInMemoryKeyValueStoreOptions& options)
      : encoder_(options),
        device_index_(-1),
        embedding_vec_size_(options.embedding_vec_size),
        num_embeddings_(options.num_embeddings),
        num_device_embeddings_(options.num_device_embeddings) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    const size_t device_embeddings_size =
        num_device_embeddings_ * embedding_vec_size_ * sizeof(Elem);
    if (device_embeddings_size > 0) {
      OF_CUDA_CHECK(cudaMalloc(&device_embeddings_, device_embeddings_size));
    }
    CHECK_GE(num_embeddings_, num_device_embeddings_);
    const size_t host_embeddings_size =
        (num_embeddings_ - num_device_embeddings_) * embedding_vec_size_ * sizeof(Elem);
    if (host_embeddings_size > 0) {
      OF_CUDA_CHECK(NumaAwareCudaMallocHost(
          device_index_, reinterpret_cast<void**>(&host_embeddings_), host_embeddings_size));
    }
  }
  ~KeyValueStoreImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(device_embeddings_));
    OF_CUDA_CHECK(cudaFreeHost(host_embeddings_));
  }

  void Prefetch(ep::Stream* stream, uint32_t num_keys, const void* keys,
                uint64_t* context) override;
  void Lookup(ep::Stream* stream, uint32_t num_keys, const void* keys, const uint64_t* context,
              void* values) override;
  void Update(ep::Stream* stream, uint32_t num_keys, const void* keys, const uint64_t* context,
              const void* values) override;

 private:
  Encoder encoder_;
  int device_index_;
  uint32_t embedding_vec_size_;
  uint64_t num_embeddings_;
  uint64_t num_device_embeddings_;
  Elem* device_embeddings_;
  Elem* host_embeddings_;
};

template<typename Encoder, typename Key, typename Elem>
void KeyValueStoreImpl<Encoder, Key, Elem>::Prefetch(ep::Stream* stream, uint32_t num_keys,
                                                     const void* keys, uint64_t* context) {
  encoder_.Encode(stream, num_keys, keys, context);
}

template<typename Encoder, typename Key, typename Elem>
void KeyValueStoreImpl<Encoder, Key, Elem>::Lookup(ep::Stream* stream, uint32_t num_keys,
                                                   const void* keys, const uint64_t* context,
                                                   void* values) {
  const uint32_t values_elem_cnt = num_keys * embedding_vec_size_;
  RUN_CUDA_KERNEL((LookupKernel<Elem>), stream, values_elem_cnt, embedding_vec_size_,
                  num_embeddings_, num_device_embeddings_, device_embeddings_, host_embeddings_,
                  values_elem_cnt, context, static_cast<Elem*>(values));
}

template<typename Encoder, typename Key, typename Elem>
void KeyValueStoreImpl<Encoder, Key, Elem>::Update(ep::Stream* stream, uint32_t num_keys,
                                                   const void* keys, const uint64_t* context,
                                                   const void* values) {
  const uint32_t values_elem_cnt = num_keys * embedding_vec_size_;
  RUN_CUDA_KERNEL((UpdateKernel<Elem>), stream, values_elem_cnt, embedding_vec_size_,
                  num_embeddings_, num_device_embeddings_, device_embeddings_, host_embeddings_,
                  values_elem_cnt, context, static_cast<const Elem*>(values));
}

}  // namespace

std::unique_ptr<KeyValueStore> NewCudaInMemoryKeyValueStore(
    const CudaInMemoryKeyValueStoreOptions& options) {
  if (options.encoding_type == CudaInMemoryKeyValueStoreOptions::EncodingType::kPlain) {
    return std::unique_ptr<KeyValueStore>(
        new KeyValueStoreImpl<PlainEncoder<int64_t>, int64_t, float>(options));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace embedding

}  // namespace oneflow
