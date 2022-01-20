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
#include "oneflow/core/embedding/hash_functions.cuh"
#include "oneflow/core/cuda/atomic.cuh"

namespace oneflow {

namespace {

// ref from
// https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/HugeCTR/src/inference/unique_op/unique_op.cu

template<typename K, typename IDX>
__global__ void UniqueIds(const int64_t capacity, const K empty_key, const IDX empty_val,
                          const int32_t num_ids, const K* ids, const IDX* column_ids, K* table_keys,
                          IDX* table_vals, IDX* num_unique_ids, K* unique_ids,
                          IDX* unique_column_ids, IDX* reverse_index) {
  CUDA_1D_KERNEL_LOOP(i, num_ids) {
    K target_key = ids[i];
    int64_t hash_index = XXH64()(target_key) % capacity;
    int64_t counter = 0;
    while (true) {
      if (counter >= capacity) {
        printf("counter >= capacity\n");
        // assert(false && "error: unique op fails: hashtable is full");
      }
      const K old_key = cuda::atomic::CAS(table_keys + hash_index, empty_key, target_key);
      volatile IDX& target_val_pos = table_vals[hash_index];
      if (empty_key == old_key) {
        IDX unique_pos = cuda::atomic::Add(num_unique_ids, 1);
        reverse_index[i] = unique_pos;
        target_val_pos = unique_pos;
        unique_ids[unique_pos] = target_key;
        unique_column_ids[unique_pos] = column_ids[i];
        break;
      } else if (target_key == old_key) {
        while (target_val_pos == empty_val)
          ;  // dead lock on sm<70
        reverse_index[i] = target_val_pos;
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
  size_t hash_table_keys_size = GetCudaAlignedSize(capacity * sizeof(K));
  size_t hash_table_vals_size = GetCudaAlignedSize(capacity * sizeof(IDX));
  return hash_table_keys_size + hash_table_vals_size;
}

template<typename K, typename IDX>
void UniqueKeys(ep::Stream* stream, const int64_t num_ids, const int64_t capacity, const K* ids,
                const IDX* column_ids, IDX* num_unique_ids, IDX* reverse_index, K* unique_ids,
                IDX* unique_column_ids, char* workspace, size_t workspace_size) {
  size_t hash_table_keys_size = GetCudaAlignedSize(capacity * sizeof(K));
  size_t hash_table_vals_size = GetCudaAlignedSize(capacity * sizeof(IDX));
  CHECK_GE(workspace_size, hash_table_keys_size + hash_table_vals_size);
  K* table_keys = reinterpret_cast<K*>(workspace);
  IDX* table_vals = reinterpret_cast<IDX*>(workspace + hash_table_keys_size);
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  K empty_key = 0;
  IDX empty_val = -1;
  cudaMemsetAsync(table_keys, 0, capacity * sizeof(K), cuda_stream);
  std::unique_ptr<ep::primitive::Fill> fill_primitive =
      ep::primitive::NewPrimitive<ep::primitive::FillFactory>(DeviceType::kCUDA,
                                                              GetDataType<IDX>::value);
  CHECK(fill_primitive);
  fill_primitive->Launch(stream, table_vals, Scalar(-1), capacity);
  cudaMemsetAsync(num_unique_ids, 0, sizeof(IDX), cuda_stream);

  UniqueIds<K, IDX><<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
      capacity, empty_key, empty_val, num_ids, ids, column_ids, table_keys, table_vals,
      num_unique_ids, unique_ids, unique_column_ids, reverse_index);
}

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_UNIQUE_KEY_H_
