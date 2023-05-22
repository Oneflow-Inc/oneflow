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

template<typename Key, typename Index, bool dump_dirty_only>
__device__ bool TryGetOrInsert(Key* entry_key, volatile Index* entry_index, bool* entry_dirty_flag,
                               Index* table_size, Key key, Index* out) {
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
      if (dump_dirty_only) {
        bool entry_flag_val = *entry_dirty_flag;
        if (!entry_flag_val) { *entry_dirty_flag = true; }
      }
      return true;
    } else if (old_entry_key == key_hi) {
      const Index entry_index_val = *entry_index;
      if (entry_index_val == 0) {
        // do nothing
      } else if ((entry_index_val & 0x1) == key_lo) {
        *out = (entry_index_val >> 1U);
        if (dump_dirty_only) {
          bool entry_flag_val = *entry_dirty_flag;
          if (!entry_flag_val) { *entry_dirty_flag = true; }
        }
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

template<typename Key, typename Index, bool dump_dirty_only>
__device__ bool GetOrInsertOne(const size_t capacity, Key* table_keys, Index* table_indices,
                               bool* table_dirty_flags, Index* table_size, Key key, size_t hash,
                               Index* out) {
  const size_t start_idx = hash % capacity;
  for (size_t count = 0; count < capacity; ++count) {
    const size_t idx = (start_idx + count) % capacity;
    Key* entry_key = table_keys + idx;
    Index* entry_index = table_indices + idx;
    bool* entry_dirty_flag = dump_dirty_only ? table_dirty_flags + idx : nullptr;
    if (TryGetOrInsert<Key, Index, dump_dirty_only>(entry_key, entry_index, entry_dirty_flag,
                                                    table_size, key, out)) {
      return true;
    }
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

template<typename Key, typename Index, bool dump_dirty_only>
__global__ void OrdinalEncodeKernel(uint64_t capacity, Key* table_keys, Index* table_indices,
                                    bool* table_dirty_flags, Index* table_size, uint32_t num_keys,
                                    const Key* keys, Index* context) {
  CUDA_1D_KERNEL_LOOP(i, num_keys) {
    Key key = keys[i];
    uint64_t hash = FullCacheHash()(key);
    bool success = GetOrInsertOne<Key, Index, dump_dirty_only>(
        capacity, table_keys, table_indices, table_dirty_flags, table_size, key, hash, context + i);
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

template<typename Key, typename Index, bool dump_dirty_only>
__global__ void OrdinalEncodeDumpKernel(const Key* table_keys, const Index* table_indices,
                                        const bool* table_dirty_flags, uint64_t start_key_index,
                                        uint64_t end_key_index, uint32_t* n_dumped, Key* keys,
                                        Index* context) {
  CUDA_1D_KERNEL_LOOP(i, (end_key_index - start_key_index)) {
    Key entry_key = table_keys[i + start_key_index];
    Index entry_index = table_indices[i + start_key_index];
    bool dump_flag = (entry_index != 0);
    if (dump_dirty_only) {
      bool entry_dirty_flag = table_dirty_flags[i + start_key_index];
      dump_flag = (dump_flag && entry_dirty_flag);
    }
    if (dump_flag) {
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

template<typename Key, typename Elem, typename Index, uint32_t block_size>
__global__ void EncodeLookupKernel(uint32_t value_length, const Elem* cache_values,
                                   uint32_t values_elem_cnt, const Key* keys, const Index* context,
                                   Elem* values, uint32_t* n_missing, Key* missing_keys,
                                   uint32_t* missing_indices, const size_t capacity,
                                   Key* table_keys, Index* table_indices) {
  constexpr uint32_t warp_size = 32;
  constexpr uint32_t n_warp_per_block = block_size / warp_size;
  const uint32_t warp_id = threadIdx.x / warp_size;
  const uint32_t lane_id = threadIdx.x % warp_size;
  const uint32_t global_warp_id = blockIdx.x * n_warp_per_block + warp_id;
  const uint32_t global_n_warp = gridDim.x * n_warp_per_block;
  const uint32_t n_keys = values_elem_cnt / value_length;
  __shared__ Key batch_keys[n_warp_per_block][warp_size];
  __shared__ Index batch_row_ids[n_warp_per_block][warp_size];
  __shared__ Key batch_missing_keys[n_warp_per_block][warp_size];
  __shared__ uint32_t batch_missing_indices[n_warp_per_block][warp_size];
  __shared__ uint32_t batch_n_missing[n_warp_per_block];
  for (uint32_t batch_start = global_warp_id * warp_size; batch_start < n_keys;
       batch_start += global_n_warp * warp_size) {
    const uint32_t batch_n_key = min(n_keys - batch_start, warp_size);
    if (lane_id == 0) { batch_n_missing[warp_id] = 0; }
    __syncwarp();
    const uint32_t key_offset = batch_start + lane_id;
    if (key_offset < n_keys) {
      const Key key = keys[batch_start + lane_id];
      const uint64_t hash = FullCacheHash()(key);
      Index row;
      GetOne<Key, Index>(capacity, table_keys, table_indices, key, hash, &row);
      batch_row_ids[warp_id][lane_id] = row;
      if (row == 0) {
        const uint32_t batch_missing_idx = atomicAdd(batch_n_missing + warp_id, 1);
        batch_missing_keys[warp_id][batch_missing_idx] = key;
        batch_missing_indices[warp_id][batch_missing_idx] = key_offset;
      }
    }
    __syncwarp();
    const uint32_t batch_n_missing_t = batch_n_missing[warp_id];
    if (lane_id == 0) {
      const uint32_t old_n_missing =
          cuda::atomic::Add(n_missing, static_cast<uint32_t>(batch_n_missing_t));
      batch_n_missing[warp_id] = old_n_missing;
    }
    __syncwarp();
    if (lane_id < batch_n_missing_t) {
      missing_keys[batch_n_missing[warp_id] + lane_id] = batch_missing_keys[warp_id][lane_id];
      missing_indices[batch_n_missing[warp_id] + lane_id] = batch_missing_indices[warp_id][lane_id];
    }
    for (int i = 0; i < batch_n_key; ++i) {
      const Key key = batch_keys[warp_id][i];
      const int64_t row = batch_row_ids[warp_id][i];
      if (row == 0) { continue; }
      for (int col = lane_id; col < value_length; col += warp_size) {
        values[(batch_start + i) * value_length + col] =
            cache_values[(row - 1) * value_length + col];
      }
    }
    __syncwarp();
  }
}

template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename Key, typename Elem, typename Index, uint32_t block_size, uint32_t pack_size>
__global__ void EncodeLookupMaskKernel(uint32_t value_length, const Elem* __restrict__ cache_values,
                                       uint32_t values_elem_cnt, const Key* __restrict__ keys,
                                       const Index* __restrict__ context, Elem* __restrict__ values,
                                       uint8_t* __restrict__ mask, const size_t capacity,
                                       Key* __restrict__ table_keys,
                                       Index* __restrict__ table_indices) {
  const uint32_t packed_cols = value_length / pack_size;
  auto* packed_values = reinterpret_cast<Pack<Elem, pack_size>*>(values);
  const auto* packed_cache_values = reinterpret_cast<const Pack<Elem, pack_size>*>(cache_values);
  constexpr uint32_t warp_size = 32;
  constexpr uint32_t n_warp_per_block = block_size / warp_size;
  const uint32_t warp_id = threadIdx.x / warp_size;
  const uint32_t lane_id = threadIdx.x % warp_size;
  const uint32_t global_warp_id = blockIdx.x * n_warp_per_block + warp_id;
  const uint32_t global_n_warp = gridDim.x * n_warp_per_block;
  const uint32_t n_keys = values_elem_cnt / value_length;
  __shared__ Key batch_keys[n_warp_per_block][warp_size];
  __shared__ Index batch_row_ids[n_warp_per_block][warp_size];
  for (uint32_t batch_start = global_warp_id * warp_size; batch_start < n_keys;
       batch_start += global_n_warp * warp_size) {
    const uint32_t batch_n_key = min(n_keys - batch_start, warp_size);
    const uint32_t key_offset = batch_start + lane_id;
    if (key_offset < n_keys) {
      const Key key = keys[batch_start + lane_id];
      const uint64_t hash = FullCacheHash()(key);
      Index row;
      GetOne<Key, Index>(capacity, table_keys, table_indices, key, hash, &row);
      batch_row_ids[warp_id][lane_id] = row;
      mask[key_offset] = row > 0;
    }
    __syncwarp();
    for (int i = 0; i < batch_n_key; ++i) {
      const Key key = batch_keys[warp_id][i];
      const int64_t row = batch_row_ids[warp_id][i];
      if (row == 0) { continue; }
#pragma unroll 4
      for (int col = lane_id; col < packed_cols; col += warp_size) {
        packed_values[(batch_start + i) * packed_cols + col] =
            packed_cache_values[(row - 1) * packed_cols + col];
      }
    }
    __syncwarp();
  }
}

template<typename Elem, typename Index, size_t pack_size>
__global__ void UpdateKernel(uint32_t value_length, Elem* cache_values, uint32_t values_elem_cnt,
                             const Index* context, const Elem* values) {
  const int packed_values_elem_cnt = values_elem_cnt / pack_size;
  const uint32_t packed_elem_cnt = value_length / pack_size;
  auto* packed_cache_values = reinterpret_cast<Pack<Elem, pack_size>*>(cache_values);
  auto* packed_values = reinterpret_cast<const Pack<Elem, pack_size>*>(values);
  CUDA_1D_KERNEL_LOOP(i, packed_values_elem_cnt) {
    const uint64_t key_id = i / packed_elem_cnt;
    const uint64_t ctx = context[key_id];
    if (ctx == 0) { continue; }
    const uint64_t row_id = ctx - 1;
    const uint64_t col_id = i - key_id * packed_elem_cnt;
    packed_cache_values[row_id * packed_elem_cnt + col_id] = packed_values[i];
  }
}

template<typename Elem, typename Index, size_t pack_size>
__global__ typename std::enable_if<std::is_same<Elem, float>::value, void>::type
FusedHalfUpdateKernel(uint32_t value_length, Elem* __restrict__ cache_values,
                      uint32_t values_elem_cnt, const Index* __restrict__ context,
                      const Elem* __restrict__ values, const half* __restrict__ update,
                      const float* __restrict__ lr, float scale) {
  const int packed_values_elem_cnt = values_elem_cnt / pack_size;
  const uint32_t packed_elem_cnt = value_length / pack_size;
  auto* packed_cache_values = reinterpret_cast<Pack<Elem, pack_size>*>(cache_values);
  auto* packed_values = reinterpret_cast<const Pack<Elem, pack_size>*>(values);
  auto* packed_update = reinterpret_cast<const Pack<half, pack_size>*>(update);
  const float alpha = -*lr * scale;
  CUDA_1D_KERNEL_LOOP(i, packed_values_elem_cnt) {
    const uint64_t key_id = i / packed_elem_cnt;
    const uint64_t ctx = context[key_id];
    if (ctx == 0) { continue; }
    const uint64_t row_id = ctx - 1;
    const uint64_t col_id = i - key_id * packed_elem_cnt;
    Pack<Elem, pack_size> m = packed_values[i];
    Pack<half, pack_size> u = packed_update[i];
    for (size_t j = 0; j < pack_size; ++j) { m.elem[j] += static_cast<Elem>(u.elem[j]) * alpha; }
    packed_cache_values[row_id * packed_elem_cnt + col_id] = m;
  }
}

template<typename Elem, typename Index, size_t pack_size>
__global__ typename std::enable_if<!std::is_same<Elem, float>::value, void>::type
FusedHalfUpdateKernel(uint32_t value_length, Elem* cache_values, uint32_t values_elem_cnt,
                      const Index* context, const Elem* values, const half* update, const float* lr,
                      float scale) {
  __trap();
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
  explicit OrdinalEncoder(uint64_t capacity, float load_factor, bool if_dump_dirty)
      : capacity_(capacity),
        table_capacity_(capacity / load_factor),
        if_dump_dirty_(if_dump_dirty) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    OF_CUDA_CHECK(cudaMalloc(&table_size_, sizeof(Index)));
    OF_CUDA_CHECK(cudaMallocHost(&table_size_host_, sizeof(Index)));
    OF_CUDA_CHECK(cudaMalloc(&table_keys_, table_capacity_ * sizeof(Key)));
    OF_CUDA_CHECK(cudaMalloc(&table_indices_, table_capacity_ * sizeof(Index)));
    if (if_dump_dirty_) {
      OF_CUDA_CHECK(cudaMalloc(&table_dirty_flags_, table_capacity_ * sizeof(bool)));
    }
    Clear();
  }
  ~OrdinalEncoder() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(table_size_));
    OF_CUDA_CHECK(cudaFreeHost(table_size_host_));
    OF_CUDA_CHECK(cudaFree(table_keys_));
    OF_CUDA_CHECK(cudaFree(table_indices_));
    if (if_dump_dirty_) { OF_CUDA_CHECK(cudaFree(table_dirty_flags_)); }
  }

  template<bool insert, bool dump_dirty_only>
  void Encode(ep::Stream* stream, uint32_t num_keys, const Key* keys, Index* context) {
    if (insert) {
      RUN_CUDA_KERNEL((OrdinalEncodeKernel<Key, Index, dump_dirty_only>), stream, num_keys,
                      table_capacity_, table_keys_, table_indices_, table_dirty_flags_, table_size_,
                      num_keys, keys, context);
    } else {
      RUN_CUDA_KERNEL((OrdinalEncodeLookupKernel<Key, Index>), stream, num_keys, table_capacity_,
                      table_keys_, table_indices_, num_keys, keys, context);
    }
  }

  void Dump(ep::Stream* stream, uint64_t start_key_index, uint64_t end_key_index,
            uint32_t* n_dumped, Key* keys, Index* context) {
    OF_CUDA_CHECK(cudaMemsetAsync(n_dumped, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
    RUN_CUDA_KERNEL((OrdinalEncodeDumpKernel<Key, Index, false>), stream,
                    end_key_index - start_key_index, table_keys_, table_indices_,
                    table_dirty_flags_, start_key_index, end_key_index, n_dumped, keys, context);
  }

  void DumpDirtyOnly(ep::Stream* stream, uint64_t start_key_index, uint64_t end_key_index,
                     uint32_t* n_dumped, Key* keys, Index* context) {
    OF_CUDA_CHECK(cudaMemsetAsync(n_dumped, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
    RUN_CUDA_KERNEL((OrdinalEncodeDumpKernel<Key, Index, true>), stream,
                    end_key_index - start_key_index, table_keys_, table_indices_,
                    table_dirty_flags_, start_key_index, end_key_index, n_dumped, keys, context);
  }

  void ClearDirtyFlags() {
    if (if_dump_dirty_) {
      OF_CUDA_CHECK(cudaMemset(table_dirty_flags_, 0, table_capacity_ * sizeof(bool)));
    }
  }

  void Clear() {
    OF_CUDA_CHECK(cudaMemset(table_size_, 0, sizeof(Index)));
    OF_CUDA_CHECK(cudaMemset(table_keys_, 0, table_capacity_ * sizeof(Key)));
    OF_CUDA_CHECK(cudaMemset(table_indices_, 0, table_capacity_ * sizeof(Index)));
    if (if_dump_dirty_) {
      OF_CUDA_CHECK(cudaMemset(table_dirty_flags_, 0, table_capacity_ * sizeof(bool)));
    }
  }

  uint64_t TableCapacity() const { return table_capacity_; }

  Key* table_keys() const { return table_keys_; }

  Index* table_indices() const { return table_indices_; }

 private:
  int device_index_{};
  Key* table_keys_;
  Index* table_indices_;
  bool* table_dirty_flags_;
  uint64_t capacity_;
  uint64_t table_capacity_;
  bool if_dump_dirty_;
  Index* table_size_{};
  Index* table_size_host_{};
};

template<typename Key, typename Elem, typename Index, size_t pack_size>
class CacheImpl : public Cache {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CacheImpl);
  explicit CacheImpl(const CacheOptions& options)
      : if_dump_dirty_(ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DUMP_DIRTY_ONLY", false)),
        encoder_(options.capacity, options.load_factor, if_dump_dirty_),
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

  DataType ValueType() const override { return options_.value_type; }

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

  void Get(ep::Stream* stream, uint32_t n_keys, const void* keys, void* values,
           uint8_t* mask) override;

  void Put(ep::Stream* stream, uint32_t n_keys, const void* keys, const void* values,
           uint32_t* n_evicted, void* evicted_keys, void* evicted_values) override;

  void FusedHalfUpdatePut(ep::Stream* stream, uint32_t n_keys, const void* keys, const void* values,
                          const void* update, const float* lr, float scale, uint32_t* n_evicted,
                          void* evicted_keys, void* evicted_values) override;
  void Dump(ep::Stream* stream, uint64_t start_key_index, uint64_t end_key_index,
            uint32_t* n_dumped, void* keys, void* values) override;

  void ClearDirtyFlags() override;

  void Clear() override;

 private:
  bool if_dump_dirty_;
  OrdinalEncoder<Key, Index> encoder_;
  int device_index_;
  uint32_t num_elem_per_value_{};
  Elem* values_;
  Index* encoding_buffer_{};
  CacheOptions options_;
  uint32_t max_query_length_;
};

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::Test(ep::Stream* stream, uint32_t n_keys,
                                                  const void* keys, uint32_t* n_missing,
                                                  void* missing_keys, uint32_t* missing_indices) {
  OF_CUDA_CHECK(
      cudaMemsetAsync(n_missing, 0, sizeof(uint32_t), stream->As<ep::CudaStream>()->cuda_stream()));
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  if (if_dump_dirty_) {
    encoder_.template Encode<false, true>(stream, n_keys, static_cast<const Key*>(keys),
                                          encoding_buffer_);
  } else {
    encoder_.template Encode<false, false>(stream, n_keys, static_cast<const Key*>(keys),
                                           encoding_buffer_);
  }
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  RUN_CUDA_KERNEL((LookupKernel<Key, Elem, Index, false>), stream, values_elem_cnt,
                  num_elem_per_value_, values_, values_elem_cnt, static_cast<const Key*>(keys),
                  encoding_buffer_, nullptr, n_missing, static_cast<Key*>(missing_keys),
                  missing_indices);
}

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::Get(ep::Stream* stream, uint32_t n_keys,
                                                 const void* keys, void* values,
                                                 uint32_t* n_missing, void* missing_keys,
                                                 uint32_t* missing_indices) {
  OF_CUDA_CHECK(
      cudaMemsetAsync(n_missing, 0, sizeof(uint32_t), stream->As<ep::CudaStream>()->cuda_stream()));
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  constexpr uint32_t block_size = 128;
  uint32_t grid_size = (n_keys + block_size - 1) / block_size;
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  EncodeLookupKernel<Key, Elem, Index, block_size>
      <<<grid_size, block_size, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          num_elem_per_value_, values_, values_elem_cnt, static_cast<const Key*>(keys),
          encoding_buffer_, static_cast<Elem*>(values), n_missing, static_cast<Key*>(missing_keys),
          missing_indices, encoder_.TableCapacity(), encoder_.table_keys(),
          encoder_.table_indices());
}

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::Get(ep::Stream* stream, uint32_t n_keys,
                                                 const void* keys, void* values, uint8_t* mask) {
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  constexpr uint32_t block_size = 128;
  uint32_t grid_size = (n_keys + block_size - 1) / block_size;
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  EncodeLookupMaskKernel<Key, Elem, Index, block_size, pack_size>
      <<<grid_size, block_size, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          num_elem_per_value_, values_, values_elem_cnt, static_cast<const Key*>(keys),
          encoding_buffer_, static_cast<Elem*>(values), mask, encoder_.TableCapacity(),
          encoder_.table_keys(), encoder_.table_indices());
}

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::Put(ep::Stream* stream, uint32_t n_keys,
                                                 const void* keys, const void* values,
                                                 uint32_t* n_evicted, void* evicted_keys,
                                                 void* evicted_values) {
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  if (if_dump_dirty_) {
    encoder_.template Encode<true, true>(stream, n_keys, static_cast<const Key*>(keys),
                                         encoding_buffer_);
  } else {
    encoder_.template Encode<true, false>(stream, n_keys, static_cast<const Key*>(keys),
                                          encoding_buffer_);
  }
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  RUN_CUDA_KERNEL((UpdateKernel<Elem, Index, pack_size>), stream, values_elem_cnt / pack_size,
                  num_elem_per_value_, values_, values_elem_cnt, encoding_buffer_,
                  static_cast<const Elem*>(values));
}

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::FusedHalfUpdatePut(
    ep::Stream* stream, uint32_t n_keys, const void* keys, const void* values, const void* update,
    const float* lr, float scale, uint32_t* n_evicted, void* evicted_keys, void* evicted_values) {
  if (!std::is_same<Elem, float>::value) { UNIMPLEMENTED(); }
  if (n_keys == 0) { return; }
  CHECK_LE(n_keys, max_query_length_);
  if (if_dump_dirty_) {
    encoder_.template Encode<true, true>(stream, n_keys, static_cast<const Key*>(keys),
                                         encoding_buffer_);
  } else {
    encoder_.template Encode<true, false>(stream, n_keys, static_cast<const Key*>(keys),
                                          encoding_buffer_);
  }
  const uint32_t values_elem_cnt = n_keys * num_elem_per_value_;
  RUN_CUDA_KERNEL((FusedHalfUpdateKernel<Elem, Index, pack_size>), stream,
                  values_elem_cnt / pack_size, num_elem_per_value_, values_, values_elem_cnt,
                  encoding_buffer_, static_cast<const Elem*>(values),
                  static_cast<const half*>(update), lr, scale);
}

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::Dump(ep::Stream* stream, uint64_t start_key_index,
                                                  uint64_t end_key_index, uint32_t* n_dumped,
                                                  void* keys, void* values) {
  if (if_dump_dirty_) {
    encoder_.DumpDirtyOnly(stream, start_key_index, end_key_index, n_dumped,
                           static_cast<Key*>(keys), encoding_buffer_);
  } else {
    encoder_.Dump(stream, start_key_index, end_key_index, n_dumped, static_cast<Key*>(keys),
                  encoding_buffer_);
  }
  RUN_CUDA_KERNEL((DumpValueKernel<Key, Elem, Index>), stream,
                  num_elem_per_value_ * (end_key_index - start_key_index), num_elem_per_value_,
                  n_dumped, encoding_buffer_, values_, static_cast<Elem*>(values));
}

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::ClearDirtyFlags() {
  encoder_.ClearDirtyFlags();
}

template<typename Key, typename Elem, typename Index, size_t pack_size>
void CacheImpl<Key, Elem, Index, pack_size>::Clear() {
  encoder_.Clear();
}

template<typename Key, typename Index>
std::unique_ptr<Cache> DispatchValueType(const CacheOptions& options) {
  if (options.value_type == DataType::kFloat) {
    const size_t value_elem_cnt = options.value_size / sizeof(float);
    const size_t half_warp = 16;
    if (value_elem_cnt % 4 == 0 && value_elem_cnt / 4 > half_warp) {
      return std::unique_ptr<Cache>(new CacheImpl<Key, float, Index, 4>(options));
    } else if (value_elem_cnt % 2 == 0 && value_elem_cnt / 2 > half_warp) {
      return std::unique_ptr<Cache>(new CacheImpl<Key, float, Index, 2>(options));
    } else {
      return std::unique_ptr<Cache>(new CacheImpl<Key, float, Index, 1>(options));
    }
  } else if (options.value_size % sizeof(ulonglong2) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, ulonglong2, Index, 1>(options));
  } else if (options.value_size % sizeof(uint64_t) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint64_t, Index, 1>(options));
  } else if (options.value_size % sizeof(uint32_t) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint32_t, Index, 1>(options));
  } else if (options.value_size % sizeof(uint16_t) == 0) {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint16_t, Index, 1>(options));
  } else {
    return std::unique_ptr<Cache>(new CacheImpl<Key, uint8_t, Index, 1>(options));
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
