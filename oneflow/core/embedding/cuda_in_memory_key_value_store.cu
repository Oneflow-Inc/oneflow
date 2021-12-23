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

template<typename Key, bool insert>
__global__ void PlainEncodingKernel(Key num_shards, uint32_t num_keys, const Key* keys,
                                    uint8_t* valid, uint64_t* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    const Key key = keys[i];
    const uint64_t idx = key / num_shards + 1;
    uint64_t ctx = idx;
    if (valid[idx] == 0) {
      if (insert) {
        valid[idx] = 1;
      } else {
        ctx = 0;
      }
    }
    context[i] = ctx;
  }
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

__device__ __inline__ int32_t AtomicCAS(uint32_t* address, uint32_t compare, uint32_t val) {
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
__device__ bool GetOne(const size_t capacity, TableEntry<Key, Index>* table, Key key, size_t hash,
                       uint64_t* out) {
  const size_t start_idx = hash % capacity;
  for (size_t count = 0; count < capacity; ++count) {
    const size_t idx = (start_idx + count) % capacity;
    TableEntry<Key, Index> entry = table[idx];
    if (entry.key == 0) { break; }
    if (entry.key == key) {
      *out = entry.index;
      return true;
    }
  }
  return false;
}

template<typename Key, typename Index>
__global__ void OrdinalEncodeKernel(uint64_t capacity, TableEntry<Key, Index>* table,
                                    uint64_t* table_size, uint32_t num_keys, const Key* keys,
                                    uint64_t* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    Key key = keys[i];
    uint64_t hash = XXH64()(key);
    bool success = GetOrInsertOne<Key, Index>(capacity, table, table_size, key, hash, context + i);
    assert(success);
  }
}

template<typename Key, typename Index>
__global__ void OrdinalEncodeLookupKernel(uint64_t capacity, TableEntry<Key, Index>* table,
                                          uint32_t num_keys, const Key* keys, uint64_t* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    Key key = keys[i];
    uint64_t hash = XXH64()(key);
    bool success = GetOne<Key, Index>(capacity, table, key, hash, context + i);
    assert(success);
  }
}

template<typename Key, typename Elem>
__global__ void LookupKernel(uint32_t value_length, uint64_t num_keys, uint64_t num_device_keys,
                             const Elem* device_values, const Elem* host_values,
                             uint32_t values_elem_cnt, const Key* keys, const uint64_t* context,
                             Elem* values, uint32_t* n_missing, Key* missing_keys,
                             uint32_t* missing_indices) {
  CUDA_1D_KERNEL_LOOP(i, values_elem_cnt) {
    const uint64_t key_id = i / value_length;
    const uint64_t ctx = context[key_id];
    const uint64_t row_id = ctx - 1;
    const uint64_t col_id = i - key_id * value_length;
    if (ctx == 0) {
      const Key missing_key = keys[key_id];
      if (missing_key == 0) {
        values[col_id] = 0;
      } else if (col_id == 0) {
        const uint32_t old_n_missing = atomicAdd(n_missing, 1);
        missing_keys[old_n_missing] = missing_key;
        missing_indices[old_n_missing] = key_id;
      }
      continue;
    }
    Elem elem;
    if (row_id < num_device_keys) {
      elem = device_values[row_id * value_length + col_id];
    } else if (row_id < num_keys) {
      elem = host_values[(row_id - num_device_keys) * value_length + col_id];
    } else {
      elem = 0;
    }
    values[i] = elem;
  }
}

template<typename Elem>
__global__ void UpdateKernel(uint32_t value_length, uint64_t num_keys, uint64_t num_device_keys,
                             Elem* device_values, Elem* host_values, uint32_t values_elem_cnt,
                             const uint64_t* context, const Elem* values) {
  CUDA_1D_KERNEL_LOOP(i, values_elem_cnt) {
    const uint64_t key_id = i / value_length;
    const uint64_t ctx = context[key_id];
    if (ctx == 0) { continue; }
    const uint64_t row_id = ctx - 1;
    const uint64_t col_id = i - key_id * value_length;
    const Elem elem = values[i];
    if (row_id < num_device_keys) {
      device_values[row_id * value_length + col_id] = elem;
    } else if (row_id < num_keys) {
      host_values[(row_id - num_device_keys) * value_length + col_id] = elem;
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
      : num_shards_(options.num_shards), capacity_(options.num_keys) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMalloc(&valid_, sizeof(uint8_t) * capacity_));
    OF_CUDA_CHECK(cudaMemset(valid_, 0, sizeof(uint8_t) * capacity_));
  }
  ~PlainEncoder() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(valid_));
  }

  template<bool insert>
  void Encode(ep::Stream* stream, uint32_t num_keys, const Key* keys, uint64_t* context) {
    RUN_CUDA_KERNEL((PlainEncodingKernel<Key, insert>), stream, num_keys, num_shards_, num_keys,
                    keys, valid_, context);
  }

 private:
  int device_index_{};
  uint8_t* valid_{};
  uint32_t num_shards_;
  size_t capacity_;
};

template<typename Key, typename Index>
class OrdinalEncoder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OrdinalEncoder);
  explicit OrdinalEncoder(const CudaInMemoryKeyValueStoreOptions& options)
      : capacity_(options.num_keys) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMalloc(&table_size_, sizeof(uint64_t)));
    OF_CUDA_CHECK(cudaMalloc(&table_, capacity_ * sizeof(TableEntry<Key, Index>)));
    OF_CUDA_CHECK(cudaMemset(table_size_, 0, sizeof(uint64_t)));
    OF_CUDA_CHECK(cudaMemset(table_, 0, capacity_ * sizeof(TableEntry<Key, Index>)));
  }
  ~OrdinalEncoder() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(table_size_));
    OF_CUDA_CHECK(cudaFree(table_));
  }

  template<bool insert>
  void Encode(ep::Stream* stream, uint32_t num_keys, const Key* keys, uint64_t* context) {
    if (insert) {
      RUN_CUDA_KERNEL((OrdinalEncodeKernel<Key, uint64_t>), stream, num_keys, capacity_, table_,
                      table_size_, num_keys, keys, context);
    } else {
      RUN_CUDA_KERNEL((OrdinalEncodeLookupKernel<Key, uint64_t>), stream, num_keys, capacity_,
                      table_, num_keys, keys, context);
    }
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
        value_length_(options.value_length),
        num_keys_(options.num_keys),
        num_device_keys_(options.num_device_keys) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    const size_t device_values_size = num_device_keys_ * value_length_ * sizeof(Elem);
    if (device_values_size > 0) { OF_CUDA_CHECK(cudaMalloc(&device_values_, device_values_size)); }
    CHECK_GE(num_keys_, num_device_keys_);
    const size_t host_values_size = (num_keys_ - num_device_keys_) * value_length_ * sizeof(Elem);
    if (host_values_size > 0) {
      OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_, reinterpret_cast<void**>(&host_values_),
                                            host_values_size));
    }
  }
  ~KeyValueStoreImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    const size_t device_values_size = num_device_keys_ * value_length_ * sizeof(Elem);
    if (device_values_size > 0) { OF_CUDA_CHECK(cudaFree(device_values_)); }
    OF_CUDA_CHECK(cudaFreeHost(host_values_));
  }

  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint32_t* n_missing, void* missing_keys, uint32_t* missing_indices,
           uint64_t* context) override;
  void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values,
           uint64_t* context) override;

 private:
  Encoder encoder_;
  int device_index_;
  uint32_t value_length_;
  uint64_t num_keys_;
  uint64_t num_device_keys_;
  Elem* device_values_;
  Elem* host_values_;
};

template<typename Encoder, typename Key, typename Elem>
void KeyValueStoreImpl<Encoder, Key, Elem>::Get(ep::Stream* stream, uint32_t num_keys,
                                                const void* keys, void* values, uint32_t* n_missing,
                                                void* missing_keys, uint32_t* missing_indices,
                                                uint64_t* context) {
  OF_CUDA_CHECK(
      cudaMemsetAsync(n_missing, 0, sizeof(uint32_t), stream->As<ep::CudaStream>()->cuda_stream()));
  if (num_keys == 0) { return; }
  encoder_.template Encode<false>(stream, num_keys, static_cast<const Key*>(keys), context);
  const uint32_t values_elem_cnt = num_keys * value_length_;
  RUN_CUDA_KERNEL((LookupKernel<Key, Elem>), stream, values_elem_cnt, value_length_, num_keys_,
                  num_device_keys_, device_values_, host_values_, values_elem_cnt,
                  static_cast<const Key*>(keys), context, static_cast<Elem*>(values), n_missing,
                  static_cast<Key*>(missing_keys), missing_indices);
}

template<typename Encoder, typename Key, typename Elem>
void KeyValueStoreImpl<Encoder, Key, Elem>::Put(ep::Stream* stream, uint32_t num_keys,
                                                const void* keys, const void* values,
                                                uint64_t* context) {
  if (num_keys == 0) { return; }
  encoder_.template Encode<true>(stream, num_keys, static_cast<const Key*>(keys), context);
  const uint32_t values_elem_cnt = num_keys * value_length_;
  RUN_CUDA_KERNEL((UpdateKernel<Elem>), stream, values_elem_cnt, value_length_, num_keys_,
                  num_device_keys_, device_values_, host_values_, values_elem_cnt, context,
                  static_cast<const Elem*>(values));
}

template<typename Key, typename Elem>
std::unique_ptr<KeyValueStore> DispatchEncodingType(
    const CudaInMemoryKeyValueStoreOptions& options) {
  if (options.encoding_type == CudaInMemoryKeyValueStoreOptions::EncodingType::kPlain) {
    return std::unique_ptr<KeyValueStore>(
        new KeyValueStoreImpl<PlainEncoder<Key>, Key, Elem>(options));
  } else if (options.encoding_type == CudaInMemoryKeyValueStoreOptions::EncodingType::kOrdinal) {
    return std::unique_ptr<KeyValueStore>(
        new KeyValueStoreImpl<OrdinalEncoder<Key, uint64_t>, Key, Elem>(options));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

template<typename Key>
std::unique_ptr<KeyValueStore> DispatchValueType(const CudaInMemoryKeyValueStoreOptions& options) {
  if (options.value_type == DataType::kFloat) {
    return DispatchEncodingType<Key, float>(options);
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

std::unique_ptr<KeyValueStore> DispatchKeyType(const CudaInMemoryKeyValueStoreOptions& options) {
  if (options.key_type == DataType::kInt32) {
    return DispatchValueType<int32_t>(options);
  } else if (options.key_type == DataType::kUInt32) {
    return DispatchValueType<uint32_t>(options);
  } else if (options.key_type == DataType::kInt64) {
    return DispatchValueType<int64_t>(options);
  } else if (options.key_type == DataType::kUInt64) {
    return DispatchValueType<uint64_t>(options);
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace

std::unique_ptr<KeyValueStore> NewCudaInMemoryKeyValueStore(
    const CudaInMemoryKeyValueStoreOptions& options) {
  return DispatchKeyType(options);
}

}  // namespace embedding

}  // namespace oneflow
