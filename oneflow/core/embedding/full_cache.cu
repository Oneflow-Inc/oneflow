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
#include "oneflow/core/embedding/full_cache.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/embedding/hash_functions.cuh"
#include "oneflow/core/cuda/atomic.cuh"

namespace oneflow {

namespace embedding {

using Key32 = unsigned int;
using Key64 = unsigned long long int;
using Key128 = ulonglong2;

namespace {

template<typename Key, typename Index>
__device__ bool TryGetOrInsert(Key* entry_key, volatile Index* entry_index, Index* table_size,
                               Key key, Index* out) {
  Key key_hi = (key | 0x1);
  Key key_lo = (key & 0x1);
  Index index_plus_one = 0;
  Key old_entry_key = cuda::atomic::CAS(entry_key, static_cast<Key>(0), key_hi);
  while (index_plus_one == 0) {
    if (old_entry_key == static_cast<Key>(0)) {
      Index index = cuda::atomic::Add(table_size, static_cast<Index>(1));
      index_plus_one = index + 1;
      *entry_index = ((index_plus_one << 1U) | key_lo);
      *out = index_plus_one;
      return true;
    } else if (old_entry_key == key_hi) {
      const Index entry_index_val = *entry_index;
      if (entry_index_val == 0) {
        // do nothing
      } else if ((entry_index_val & 0x1) == key_lo) {
        *out = (entry_index_val >> 1U);
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return false;
}

template<typename Key, typename Index>
__device__ bool GetOrInsertOne(const size_t capacity, Key* table_keys, Index* table_indices,
                               Index* table_size, Key key, size_t hash, Index* out) {
  const size_t start_idx = hash % capacity;
  for (size_t count = 0; count < capacity; ++count) {
    const size_t idx = (start_idx + count) % capacity;
    Key* entry_key = table_keys + idx;
    Index* entry_index = table_indices + idx;
    if (TryGetOrInsert<Key, Index>(entry_key, entry_index, table_size, key, out)) { return true; }
  }
  return false;
}

template<typename Key, typename Index>
__device__ bool GetOne(const size_t capacity, Key* table_keys, Index* table_indices, Key key,
                       size_t hash, Index* out) {
  const size_t start_idx = hash % capacity;
  for (size_t count = 0; count < capacity; ++count) {
    const size_t idx = (start_idx + count) % capacity;
    Key entry_key = table_keys[idx];
    Key entry_index = table_indices[idx];
    Key key_hi = (key | 0x1);
    Key key_lo = (key & 0x1);
    if (entry_key == 0) { break; }
    if (entry_key == key_hi) {
      if ((entry_index & 0x1) == key_lo) {
        *out = (entry_index >> 1U);
        return true;
      }
    }
  }
  *out = 0;
  return false;
}

template<typename Key, typename Index>
__global__ void OrdinalEncodeKernel(uint64_t capacity, Key* table_keys, Index* table_indices,
                                    Index* table_size, uint32_t num_keys, const Key* keys,
                                    Index* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    Key key = keys[i];
    uint64_t hash = FullCacheHash()(key);
    bool success = GetOrInsertOne<Key, Index>(capacity, table_keys, table_indices, table_size, key,
                                              hash, context + i);
    assert(success);
  }
}

template<typename Key, typename Index>
__global__ void OrdinalEncodeLookupKernel(uint64_t capacity, Key* table_keys, Index* table_indices,
                                          uint32_t num_keys, const Key* keys, Index* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    Key key = keys[i];
    uint64_t hash = FullCacheHash()(key);
    GetOne<Key, Index>(capacity, table_keys, table_indices, key, hash, context + i);
  }
}

template<typename Key, typename Index>
__global__ void OrdinalEncodeDumpKernel(const Key* table_keys, const Index* table_indices,
                                        uint64_t start_key_index, uint64_t end_key_index,
                                        uint32_t* n_dumped, Key* keys, Index* context) {
  CUDA_1D_KERNEL_LOOP(i, (end_key_index - start_key_index)) {
    Key entry_key = table_keys[i + start_key_index];
    Index entry_index = table_indices[i + start_key_index];
    if (entry_index != 0) {
      uint32_t index = cuda::atomic::Add(n_dumped, static_cast<uint32_t>(1));
      keys[index] = ((entry_key ^ 0x1) | (entry_index & 0x1));
      context[index] = (entry_index >> 1U);
    }
  }
}

template<typename Key, typename Elem, typename Index, bool return_value>
__global__ void LookupKernel(uint32_t value_length, const Elem* cache_values,
                             uint32_t values_elem_cnt, const Key* keys, const Index* context,
                             Elem* values, uint32_t* n_missing, Key* missing_keys,
                             uint32_t* missing_indices) {
  CUDA_1D_KERNEL_LOOP(i, values_elem_cnt) {
    const uint64_t key_id = i / value_length;
    const uint64_t ctx = context[key_id];
    const uint64_t row_id = ctx - 1;
    const uint64_t col_id = i - key_id * value_length;
    if (ctx == 0) {
      const Key missing_key = keys[key_id];
      if (col_id == 0) {
        const uint32_t old_n_missing = cuda::atomic::Add(n_missing, static_cast<uint32_t>(1));
        missing_keys[old_n_missing] = missing_key;
        missing_indices[old_n_missing] = key_id;
      }
      continue;
    }
    if (return_value) { values[i] = cache_values[row_id * value_length + col_id]; }
  }
}

template<typename Elem, typename Index>
__global__ void UpdateKernel(uint32_t value_length, Elem* cache_values, uint32_t values_elem_cnt,
                             const Index* context, const Elem* values) {
  CUDA_1D_KERNEL_LOOP(i, values_elem_cnt) {
    const uint64_t key_id = i / value_length;
    const uint64_t ctx = context[key_id];
    if (ctx == 0) { continue; }
    const uint64_t row_id = ctx - 1;
    const uint64_t col_id = i - key_id * value_length;
    const Elem elem = values[i];
    cache_values[row_id * value_length + col_id] = elem;
  }
}

template<typename Key, typename Elem, typename Index>
__global__ void DumpValueKernel(uint32_t value_length, const uint32_t* n_dumped,
                                const Index* context, const Elem* cache_values, Elem* values) {
  CUDA_1D_KERNEL_LOOP(i, *n_dumped * value_length) {
    const uint64_t key_id = i / value_length;
    const uint64_t ctx = context[key_id];
    const uint64_t row_id = ctx - 1;
    const uint64_t col_id = i - key_id * value_length;
    values[i] = cache_values[row_id * value_length + col_id];
  }
}

template<typename Key, typename Index>
class OrdinalEncoder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OrdinalEncoder);
  explicit OrdinalEncoder(uint64_t capacity, float load_factor)
      : capacity_(capacity), table_capacity_(capacity / load_factor) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMalloc(&table_size_, sizeof(Index)));
    OF_CUDA_CHECK(cudaMallocHost(&table_size_host_, sizeof(Index)));
    OF_CUDA_CHECK(cudaMalloc(&table_keys_, table_capacity_ * sizeof(Key)));
    OF_CUDA_CHECK(cudaMalloc(&table_indices_, table_capacity_ * sizeof(Index)));
    Clear();
  }
  ~OrdinalEncoder() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(table_size_));
    OF_CUDA_CHECK(cudaFreeHost(table_size_host_));
    OF_CUDA_CHECK(cudaFree(table_keys_));
    OF_CUDA_CHECK(cudaFree(table_indices_));
  }

  template<bool insert>
  void Encode(ep::Stream* stream, uint32_t num_keys, const Key* keys, Index* context) {
    if (insert) {
      RUN_CUDA_KERNEL((OrdinalEncodeKernel<Key, Index>), stream, num_keys, table_capacity_,
                      table_keys_, table_indices_, table_size_, num_keys, keys, context);
      OF_CUDA_CHECK(cudaMemcpyAsync(table_size_host_, table_size_, sizeof(Index), cudaMemcpyDefault,
                                    stream->As<ep::CudaStream>()->cuda_stream()));
      CHECK_JUST(stream->Sync());
      CHECK_LT(*table_size_host_, capacity_)
          << "The number of key is larger than cache size, please enlarge cache_memory_budget. ";
    } else {
      RUN_CUDA_KERNEL((OrdinalEncodeLookupKernel<Key, Index>), stream, num_keys, table_capacity_,
                      table_keys_, table_indices_, num_keys, keys, context);
    }
  }

  void Dump(ep::Stream* stream, uint64_t start_key_index, uint64_t end_key_index,
            uint32_t* n_dumped, Key* keys, Index* context) {
    OF_CUDA_CHECK(cudaMemsetAsync(n_dumped, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
    RUN_CUDA_KERNEL((OrdinalEncodeDumpKernel<Key, Index>), stream, end_key_index - start_key_index,
                    table_keys_, table_indices_, start_key_index, end_key_index, n_dumped, keys,
                    context);
  }

  void Clear() {
    OF_CUDA_CHECK(cudaMemset(table_size_, 0, sizeof(Index)));
    OF_CUDA_CHECK(cudaMemset(table_keys_, 0, table_capacity_ * sizeof(Key)));
    OF_CUDA_CHECK(cudaMemset(table_indices_, 0, table_capacity_ * sizeof(Index)));
  }

  uint64_t TableCapacity() const { return table_capacity_; }

 private:
  int device_index_{};
  Key* table_keys_;
  Index* table_indices_;
  uint64_t capacity_;
  uint64_t table_capacity_;
  Index* table_size_{};
  Index* table_size_host_{};
};

template<typename Key, typename Elem, typename Index>
class CacheImpl : public Cache {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CacheImpl);
  explicit CacheImpl(const CacheOptions& options)
      : encoder_(options.capacity, options.load_factor),
        device_index_(-1),
        options_(options),
        max_query_length_(0) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    const uint64_t values_size = options.capacity * options.value_size;
    if (options.value_memory_kind == CacheOptions::MemoryKind::kDevice) {
      OF_CUDA_CHECK(cudaMalloc(&values_, values_size));
    } else if (options.value_memory_kind == CacheOptions::MemoryKind::kHost) {
      if (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_NUMA_AWARE_ALLOCATION", false)) {
        OF_CUDA_CHECK(cudaMallocHost(&values_, values_size));
      } else {
        OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_, reinterpret_cast<void**>(&values_),
                                              values_size));
      }
    } else {
      UNIMPLEMENTED();
    }
    num_elem_per_value_ = options_.value_size / sizeof(Elem);
  }
  ~CacheImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    if (options_.value_memory_kind == CacheOptions::MemoryKind::kDevice) {
      OF_CUDA_CHECK(cudaFree(values_));
    } else if (options_.value_memory_kind == CacheOptions::MemoryKind::kHost) {
      OF_CUDA_CHECK(cudaFreeHost(values_));
    } else {
      UNIMPLEMENTED();
    }
    if (max_query_length_ > 0) { OF_CUDA_CHECK(cudaFree(encoding_buffer_)); }
  }

  uint64_t Capacity() const override { return options_.capacity; }
  uint64_t DumpCapacity() const override { return encoder_.TableCapacity(); }
  uint32_t KeySize() const override { return options_.key_size; }

  uint32_t ValueSize() const override { return options_.value_size; }

  uint32_t MaxQueryLength() const override { return max_query_length_; }

  void ReserveQueryLength(uint32_t query_length) override {
    CudaCurrentDeviceGuard guard(device_index_);
    if (query_length <= max_query_length_) { return; }
    if (max_query_length_ > 0) { OF_CUDA_CHECK(cudaFree(encoding_buffer_)); }
    OF_CUDA_CHECK(cudaMalloc(&encoding_buffer_, query_length * sizeof(uint64_t)));
    max_query_length_ = query_length;
  }

  CacheOptions::Policy Policy() const override { return CacheOptions::Policy::kFull; }

  void Test(ep::Stream* stream, uint32_t n_keys, const void* keys, uint32_t* n_missing,
            void* missing_keys, uint32_t* missing_indices) override;

  void Get(ep::Stream* stream, uint32_t n_keys, const void* keys, void* values, uint32_t* n_missing,
           void* missing_keys, uint32_t* missing_indices) override;

  void Put(ep::Stream* stream, uint32_t n_keys, const void* keys, const void* values,
           uint32_t* n_evicted, void* evicted_keys, void* evicted_values) override;

  void Dump(ep::Stream* stream, uint64_t start_key_index, uint64_t end_key_index,
            uint32_t* n_dumped, void* keys, void* values) override;

  void Clear() override;

 private:
  OrdinalEncoder<Key, Index> encoder_;
  int device_index_;
  uint32_t num_elem_per_value_{};
  Elem* values_;
  Index* encoding_buffer_{};
  CacheOptions options_;
  uint32_t max_query_length_;
};

template<typename Key, typename Elem, typename Index>
void CacheImpl<Key, Elem, Index>::Test(ep::Stream* stream, uint32_t n_keys, const void* keys,
                                       uint32_t* n_missing, void* missing_keys,
                                       uint32_t* missing_indices) {
  OF_CUDA_CHECK(
      cudaMemsetAsync(n_missing, 0, sizeof(uint32_t), stream->As<ep::CudaStream>()->cuda_stream()));
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  encoder_.template Encode<false>(stream, n_keys, static_cast<const Key*>(keys), encoding_buffer_);
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  RUN_CUDA_KERNEL((LookupKernel<Key, Elem, Index, false>), stream, values_elem_cnt,
                  num_elem_per_value_, values_, values_elem_cnt, static_cast<const Key*>(keys),
                  encoding_buffer_, nullptr, n_missing, static_cast<Key*>(missing_keys),
                  missing_indices);
}

template<typename Key, typename Elem, typename Index>
void CacheImpl<Key, Elem, Index>::Get(ep::Stream* stream, uint32_t n_keys, const void* keys,
                                      void* values, uint32_t* n_missing, void* missing_keys,
                                      uint32_t* missing_indices) {
  OF_CUDA_CHECK(
      cudaMemsetAsync(n_missing, 0, sizeof(uint32_t), stream->As<ep::CudaStream>()->cuda_stream()));
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  encoder_.template Encode<false>(stream, n_keys, static_cast<const Key*>(keys), encoding_buffer_);
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  RUN_CUDA_KERNEL((LookupKernel<Key, Elem, Index, true>), stream, values_elem_cnt,
                  num_elem_per_value_, values_, values_elem_cnt, static_cast<const Key*>(keys),
                  encoding_buffer_, static_cast<Elem*>(values), n_missing,
                  static_cast<Key*>(missing_keys), missing_indices);
}

template<typename Key, typename Elem, typename Index>
void CacheImpl<Key, Elem, Index>::Put(ep::Stream* stream, uint32_t n_keys, const void* keys,
                                      const void* values, uint32_t* n_evicted, void* evicted_keys,
                                      void* evicted_values) {
  OF_CUDA_CHECK(
      cudaMemsetAsync(n_evicted, 0, sizeof(uint32_t), stream->As<ep::CudaStream>()->cuda_stream()));
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  encoder_.template Encode<true>(stream, n_keys, static_cast<const Key*>(keys), encoding_buffer_);
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  RUN_CUDA_KERNEL((UpdateKernel<Elem, Index>), stream, values_elem_cnt, num_elem_per_value_,
                  values_, values_elem_cnt, encoding_buffer_, static_cast<const Elem*>(values));
}

template<typename Key, typename Elem, typename Index>
void CacheImpl<Key, Elem, Index>::Dump(ep::Stream* stream, uint64_t start_key_index,
                                       uint64_t end_key_index, uint32_t* n_dumped, void* keys,
                                       void* values) {
  encoder_.Dump(stream, start_key_index, end_key_index, n_dumped, static_cast<Key*>(keys),
                encoding_buffer_);
  RUN_CUDA_KERNEL((DumpValueKernel<Key, Elem, Index>), stream,
                  num_elem_per_value_ * (end_key_index - start_key_index), num_elem_per_value_,
                  n_dumped, encoding_buffer_, values_, static_cast<Elem*>(values));
}

template<typename Key, typename Elem, typename Index>
void CacheImpl<Key, Elem, Index>::Clear() {
  encoder_.Clear();
}

template<typename Key, typename Index>
std::unique_ptr<Cache> DispatchValueType(const CacheOptions& options) {
  if (options.value_size % sizeof(ulonglong2) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, ulonglong2, Index>(options));
  } else if (options.value_size % sizeof(uint64_t) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint64_t, Index>(options));
  } else if (options.value_size % sizeof(uint32_t) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint32_t, Index>(options));
  } else if (options.value_size % sizeof(uint16_t) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint16_t, Index>(options));
  } else {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint8_t, Index>(options));
  }
}

template<typename Index>
std::unique_ptr<Cache> DispatchKeyType(const CacheOptions& options) {
  if (options.key_size == sizeof(Key32)) {
    return DispatchValueType<Key32, Index>(options);
  } else if (options.key_size == sizeof(Key64)) {
    return DispatchValueType<Key64, Index>(options);
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

std::unique_ptr<Cache> DispatchIndexType(const CacheOptions& options) {
  const int64_t table_capacity = static_cast<double>(options.capacity) / options.load_factor;
  if (table_capacity >= (1ULL << 31ULL)) {
    return DispatchKeyType<uint64_t>(options);
  } else {
    return DispatchKeyType<uint32_t>(options);
  }
}

}  // namespace

std::unique_ptr<Cache> NewFullCache(const CacheOptions& options) {
  return DispatchIndexType(options);
}

}  // namespace embedding

}  // namespace oneflow
