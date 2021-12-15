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
#include "oneflow/core/embedding/hash_functions.cuh"

namespace oneflow {

namespace embedding {

namespace {

template<typename Key>
__global__ void PlainEncodingKernel(Key num_shards, uint32_t num_keys, const Key* keys,
                                    uint64_t* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) { context[i] = keys[i] / num_shards; }
}

template<typename Key, typename Index>
struct alignas(2 * std::max(sizeof(Key), sizeof(Index))) TableEntry {
  Key key;
  Index index;
};

using CuInt64T = unsigned long long int;

__device__ __inline__ int32_t AtomicCAS(int32_t* address, int32_t compare, int32_t val) {
  return atomicCAS(address, compare, val);
}

__device__ __inline__ int64_t AtomicCAS(int64_t* address, int64_t compare, int64_t val) {
  static_assert(sizeof(int64_t) == sizeof(CuInt64T), "size error");
  return static_cast<int64_t>(atomicCAS(reinterpret_cast<CuInt64T*>(address),
                                        static_cast<CuInt64T>(compare),
                                        static_cast<CuInt64T>(val)));
}

__device__ __inline__ uint64_t AtomicCAS(uint64_t* address, uint64_t compare, uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(CuInt64T), "size error");
  return static_cast<uint64_t>(atomicCAS(reinterpret_cast<CuInt64T*>(address),
                                         static_cast<CuInt64T>(compare),
                                         static_cast<CuInt64T>(val)));
}

__device__ __inline__ int32_t AtomicAdd(int32_t* address, int32_t val) {
  return atomicAdd(address, val);
}

__device__ __inline__ int32_t AtomicAdd(uint32_t* address, uint32_t val) {
  return atomicAdd(address, val);
}

__device__ __inline__ int32_t AtomicAdd(uint64_t* address, uint64_t val) {
  return atomicAdd(reinterpret_cast<unsigned long long int*>(address), val);
}

__device__ __inline__ int64_t AtomicAdd(int64_t* address, int64_t val) {
  static_assert(sizeof(int64_t) == sizeof(CuInt64T), "size error");
  return static_cast<int64_t>(
      atomicAdd(reinterpret_cast<CuInt64T*>(address), static_cast<CuInt64T>(val)));
}

template<typename Key, typename Index>
__device__ bool TryGetOrInsert(Key* entry_key, volatile Index* entry_index, uint64_t* table_size,
                               Key key, uint64_t* out) {
  Key old_entry_key = AtomicCAS(entry_key, static_cast<Key>(0), key);
  if (old_entry_key == 0) {
    Index index = AtomicAdd(table_size, 1) + 1;
    *entry_index = index;
    *out = index;
    return true;
  } else if (old_entry_key == key) {
    while (true) {
      Index index = *entry_index;
      if (index != 0) {
        *out = index;
        break;
      }
    }
    return true;
  } else {
    return false;
  }
}

template<typename Key, typename Index>
__device__ bool GetOrInsertOne(const size_t capacity, TableEntry<Key, Index>* table,
                               uint64_t* table_size, Key key, size_t hash, uint64_t* out) {
  const size_t start_idx = hash % capacity;
  // fast path
  {
    TableEntry<Key, Index> entry = table[start_idx];
    if (entry.key == key && entry.index != 0) {
      *out = entry.index;
      return true;
    }
  }
  for (size_t count = 0; count < capacity; ++count) {
    const size_t idx = (start_idx + count) % capacity;
    Key* entry_key = &table[idx].key;
    Index* entry_index = &table[idx].index;
    if (TryGetOrInsert<Key, Index>(entry_key, entry_index, table_size, key, out)) { return true; }
  }
  return false;
}

template<typename Key, typename Index>
__global__ void OrdinalEncodingKernel(uint64_t capacity, TableEntry<Key, Index>* table,
                                      uint64_t* table_size, uint32_t num_keys, const Key* keys,
                                      uint64_t* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    Key key = keys[i];
    uint64_t hash = XXH64()(key);
    bool success = GetOrInsertOne<Key, Index>(capacity, table, table_size, key, hash, context + i);
    assert(success);
  }
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

  void Encode(ep::Stream* stream, uint32_t num_keys, const Key* keys, uint64_t* context) {
    RUN_CUDA_KERNEL((PlainEncodingKernel<Key>), stream, num_keys, num_shards_, num_keys, keys,
                    context);
  }

 private:
  uint32_t num_shards_;
};

template<typename Key, typename Index>
class OrdinalEncoder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OrdinalEncoder);
  explicit OrdinalEncoder(const CudaInMemoryKeyValueStoreOptions& options)
      : capacity_(options.num_embeddings) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMalloc(&table_size_, sizeof(uint64_t)));
    OF_CUDA_CHECK(cudaMalloc(&table_, capacity_ * sizeof(TableEntry<Key, Index>)));
  }
  ~OrdinalEncoder() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(table_size_));
    OF_CUDA_CHECK(cudaFree(table_));
  }

  void Encode(ep::Stream* stream, uint32_t num_keys, const Key* keys, uint64_t* context) {
    RUN_CUDA_KERNEL((OrdinalEncodingKernel<Key>), stream, num_keys, capacity_, table_, table_size_,
                    num_keys, keys, context);
  }

 private:
  int device_index_{};
  TableEntry<Key, Index>* table_;
  uint64_t capacity_;
  uint64_t* table_size_{};
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
  encoder_.Encode(stream, num_keys, static_cast<const Key*>(keys), context);
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
        new KeyValueStoreImpl<PlainEncoder<uint64_t>, uint64_t, float>(options));
  } else if (options.encoding_type == CudaInMemoryKeyValueStoreOptions::EncodingType::kOrdinal) {
    return std::unique_ptr<KeyValueStore>(
        new KeyValueStoreImpl<OrdinalEncoder<uint64_t, uint64_t>, uint64_t, float>(options));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace embedding

}  // namespace oneflow
