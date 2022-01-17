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

#ifndef ONEFLOW_USER_KERNELS_UNIQUE_KEY_H_
#define ONEFLOW_USER_KERNELS_UNIQUE_KEY_H_
#include "oneflow/core/ep/include/primitive/fill.h"

namespace oneflow {

namespace {

template<typename K, typename IDX>
__global__ void dump_kernel(const K* d_key, const IDX* d_key2, const IDX* keys, K* unique_keys,
                            IDX* unique_keys2, IDX* table2unique_offset, IDX* counter,
                            const size_t offset, const size_t search_length, const K empty_key) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  K read_key;
  IDX read_val;
  bool valid_slot = false;
  // Each thread gather the key and value from slot assigned to them.
  if (idx < search_length) {
    read_key = keys[offset + idx];
    if (read_key != empty_key) {
      valid_slot = true;
      K old_count = atomicAdd(counter, 1);
      unique_keys[old_count] = d_key[read_key];
      unique_keys2[old_count] = d_key2[read_key];
      table2unique_offset[idx] = old_count;
    }
  }
}

template<typename IDX>
__global__ void gather(int64_t n, IDX* index, IDX* values, IDX* out) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { out[idx] = values[index[idx]]; }
}

// template <typename K, typename IDX, typename hasher>
template<typename K, typename IDX>
__global__ void get_insert_kernel(const K* d_key, IDX* reverse_index, const size_t len, IDX* keys,
                                  const size_t capacity, const IDX empty_key) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    // K target_key = d_key[idx];
    // save id's index in hash table instead of id
    IDX target_key = idx;
    // size_t hash_index = hasher::hash(d_key[idx]) % capacity;
    size_t hash_index = d_key[idx] % capacity;
    size_t counter = 0;
    while (true) {
      // Have searched all the slot in the hashtable, but all slots in the hashtable are occupied by
      // other keys
      if (counter >= capacity) {
        printf("counter >= capacity\n");
        // assert(false && "error: unique op fails: hashtable is full");
      }
      // Try to set the key for the current slot to target key
      const IDX old_key = atomicCAS(keys + hash_index, empty_key, target_key);
      if (empty_key == old_key) {
        reverse_index[idx] = hash_index;
        break;
      } else if (d_key[target_key] == d_key[old_key]) {
        if (target_key < old_key) {
          // when same key, use lower index in hash table
          atomicCAS(keys + hash_index, old_key, target_key);
        }
        reverse_index[idx] = hash_index;
        break;
      }
      counter++;
      hash_index = (hash_index + 1) % capacity;
    }
  }
}
}  // namespace

template<typename K, typename IDX>
size_t GetUniqueKeysWorkspace(const int64_t num_ids, const int64_t capacity) {
  size_t hash_table_size = GetCudaAlignedSize(capacity * sizeof(IDX));
  size_t d_reverse_index_size = GetCudaAlignedSize(num_ids * sizeof(IDX));
  size_t table2unique_offset_size = GetCudaAlignedSize(capacity * sizeof(IDX));
  return hash_table_size + d_reverse_index_size + table2unique_offset_size;
}

template<typename K, typename IDX>
void UniqueKeys(ep::Stream* stream, const int64_t num_ids, const int64_t capacity, const K* ids,
                const IDX* slots, IDX* num_unique_ids, IDX* reverse_index, K* unique_ids,
                IDX* unique_slots, char* workspace) {
  size_t hash_table_size = GetCudaAlignedSize(capacity * sizeof(IDX));
  size_t d_reverse_index_size = GetCudaAlignedSize(num_ids * sizeof(IDX));
  size_t table2unique_offset_size = GetCudaAlignedSize(capacity * sizeof(IDX));
  IDX* table_key = reinterpret_cast<IDX*>(workspace);
  IDX* d_reverse_index = reinterpret_cast<IDX*>(workspace + hash_table_size);
  IDX* table2unique_offset =
      reinterpret_cast<IDX*>(workspace + hash_table_size + d_reverse_index_size);
  IDX empty_key = -1;
  std::unique_ptr<ep::primitive::Fill> fill_primitive =
      ep::primitive::NewPrimitive<ep::primitive::FillFactory>(DeviceType::kCUDA,
                                                              GetDataType<IDX>::value);
  CHECK(fill_primitive);
  fill_primitive->Launch(stream, table_key, Scalar(-1), capacity);
  cudaMemset(num_unique_ids, 0, sizeof(IDX));
  int32_t BLOCK_SIZE_ = 256;
  get_insert_kernel<K, IDX><<<(num_ids - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0,
                              stream->As<ep::CudaStream>()->cuda_stream()>>>(
      ids, d_reverse_index, num_ids, table_key, capacity, empty_key);
  // Launch dump kernel
  dump_kernel<K, IDX><<<(capacity - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0,
                        stream->As<ep::CudaStream>()->cuda_stream()>>>(
      ids, slots, table_key, unique_ids, unique_slots, table2unique_offset, num_unique_ids, 0,
      capacity, empty_key);
  gather<IDX><<<(capacity - 1) / BLOCK_SIZE_ + 1, BLOCK_SIZE_, 0,
                stream->As<ep::CudaStream>()->cuda_stream()>>>(capacity, d_reverse_index,
                                                               table2unique_offset, reverse_index);
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_UNIQUE_KEY_H_
