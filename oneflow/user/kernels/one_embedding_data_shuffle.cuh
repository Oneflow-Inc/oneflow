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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/gather_kernel_util.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/embedding/hash_functions.cuh"

namespace oneflow {

namespace data_shuffle {

template<typename K>
struct TableEntry {
  K key;
  uint32_t value;
};

template<typename U>
__global__ void GenerateTableIds(int32_t elem_cnt, int32_t num_tables, U* table_ids) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { table_ids[i] = i % num_tables; }
}

namespace {

constexpr uint32_t PADDING_REV_INDEX = 0xffffffff;

template<typename K, typename V, typename IDX, typename HASH>
__global__ void HashTableUniqueAndPartitionPairs(
    const uint32_t table_capacity, const uint32_t num_keys, int32_t num_partition,
    IDX* unique_counts, TableEntry<K>* table, const K* keys, const V* values,
    K* partitioned_unique_keys, V* partitioned_unique_values, IDX* reverse_index,
    bool need_process_values, const bool has_padding_idx, const int64_t padding_idx) {
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_keys) {
    IDX r_index_plus_one = 0;
    const K key = keys[i];
    if (has_padding_idx && key == padding_idx) {
      reverse_index[i] = PADDING_REV_INDEX;
    } else {
      size_t key_hash = HASH()(key);
      uint32_t partition_id = key_hash % num_partition;
      IDX* unique_count = unique_counts + partition_id;
      K* unique_keys = partitioned_unique_keys + partition_id * num_keys;
      uint32_t pos = key_hash % table_capacity;
      const K key_hi = (key | 0x1);
      const K key_lo = (key & 0x1);
      uint32_t counter = 0;
      while (r_index_plus_one == 0) {
        bool prob_next = false;
        K* key_ptr = &table[pos].key;
        volatile uint32_t* table_value_ptr = &table[pos].value;
        const K old_key = cuda::atomic::CAS(key_ptr, 0, key_hi);
        if (old_key == 0) {
          IDX unique_pos = cuda::atomic::Add(unique_count, 1);
          r_index_plus_one = unique_pos + 1;
          unique_keys[unique_pos] = key;
          if (need_process_values) {
            partitioned_unique_values[partition_id * num_keys + unique_pos] = values[i];
          }
          *table_value_ptr = ((r_index_plus_one << 1U) | key_lo);
        } else if (old_key == key_hi) {
          const uint32_t value = *table_value_ptr;
          if (value == 0) {
            // do nothing
          } else if ((value & 0x1) == key_lo) {
            r_index_plus_one = (value >> 1U);
          } else {
            prob_next = true;
          }
        } else {
          prob_next = true;
        }
        if (prob_next) {
          pos += 1;
          counter += 1;
          if (pos >= table_capacity) { pos -= table_capacity; }
          if (counter >= table_capacity) { __trap(); }
        }
      }
      reverse_index[i] = partition_id * num_keys + r_index_plus_one - 1;
    }
  }
}

template<typename IDX>
__global__ void ComputeOffset(int32_t n, IDX* value) {
  IDX sum = 0;
  for (int i = 0; i < n; ++i) {
    IDX count = value[i];
    value[i] = sum;
    sum += count;
  }
}

template<typename IDX>
__global__ void ContiguousInverseUniquePartitionIndices(const int32_t num_ids, IDX* indices_offset,
                                                        IDX* inverse_ptr) {
  CUDA_1D_KERNEL_LOOP(i, num_ids) {
    int inverse_indice = inverse_ptr[i];
    int partition_id = inverse_indice / num_ids;
    int partition_indice = inverse_indice - partition_id * num_ids;
    int new_offset = indices_offset[partition_id];
    inverse_ptr[i] = new_offset + partition_indice;
  }
}

template<typename T>
void ShuffleData(cudaStream_t cuda_stream, ncclComm_t comm, DataType data_type,
                 const std::vector<int64_t>& send_offsets,
                 const std::vector<int64_t>& send_elem_cnt, const T* send_data,
                 const std::vector<int64_t>& recv_offsets,
                 const std::vector<int64_t>& recv_elem_cnt, T* recv_data) {
  ncclDataType_t nccl_data_type = GetNcclDataType(data_type);
  const int64_t parallel_num = send_offsets.size();
  OF_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < parallel_num; ++i) {
    OF_NCCL_CHECK(ncclSend(send_data + send_offsets.at(i), send_elem_cnt.at(i), nccl_data_type, i,
                           comm, cuda_stream));
    OF_NCCL_CHECK(ncclRecv(recv_data + recv_offsets.at(i), recv_elem_cnt.at(i), nccl_data_type, i,
                           comm, cuda_stream));
  }
  OF_NCCL_CHECK(ncclGroupEnd());
}

template<typename IDX>
void MakeShuffleIdParams(const IDX* host_num_unique_matrix, const int64_t num_ids,
                         const int64_t row_size, int64_t parallel_id, int64_t parallel_num,
                         std::vector<int64_t>* scatter_offset_vec,
                         std::vector<int64_t>* scatter_elem_cnt_vec,
                         std::vector<int64_t>* gather_offset_vec,
                         std::vector<int64_t>* gather_elem_cnt_vec) {
  scatter_offset_vec->resize(parallel_num);
  scatter_elem_cnt_vec->resize(parallel_num);
  gather_offset_vec->resize(parallel_num);
  gather_elem_cnt_vec->resize(parallel_num);
  int64_t gather_offset = 0;
  for (int64_t i = 0; i < parallel_num; ++i) {
    const int64_t scatter_elem_cnt =
        host_num_unique_matrix[parallel_id * parallel_num + i] * row_size;
    const int64_t gather_elem_cnt =
        host_num_unique_matrix[i * parallel_num + parallel_id] * row_size;
    scatter_offset_vec->at(i) = i * num_ids * row_size;
    scatter_elem_cnt_vec->at(i) = scatter_elem_cnt;
    gather_offset_vec->at(i) = gather_offset;
    gather_elem_cnt_vec->at(i) = gather_elem_cnt;
    gather_offset += gather_elem_cnt;
  }
}

template<typename IDX>
void MakeShuffleParams(const IDX* host_num_unique_matrix, const int64_t num_ids,
                       const int64_t row_size, int64_t parallel_id, int64_t parallel_num,
                       std::vector<int64_t>* scatter_offset_vec,
                       std::vector<int64_t>* scatter_elem_cnt_vec,
                       std::vector<int64_t>* gather_offset_vec,
                       std::vector<int64_t>* gather_elem_cnt_vec) {
  scatter_offset_vec->resize(parallel_num);
  scatter_elem_cnt_vec->resize(parallel_num);
  gather_offset_vec->resize(parallel_num);
  gather_elem_cnt_vec->resize(parallel_num);
  int64_t gather_offset = 0;
  int64_t scatter_offset = 0;
  for (int64_t i = 0; i < parallel_num; ++i) {
    const int64_t scatter_elem_cnt =
        host_num_unique_matrix[parallel_id * parallel_num + i] * row_size;
    const int64_t gather_elem_cnt =
        host_num_unique_matrix[i * parallel_num + parallel_id] * row_size;
    scatter_offset_vec->at(i) = scatter_offset;
    scatter_elem_cnt_vec->at(i) = scatter_elem_cnt;
    gather_offset_vec->at(i) = gather_offset;
    gather_elem_cnt_vec->at(i) = gather_elem_cnt;
    scatter_offset += scatter_elem_cnt;
    gather_offset += gather_elem_cnt;
  }
}

template<typename K, typename U, typename IDX>
void ShuffleIdsAndTableIds(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                           int64_t parallel_num, int64_t num_ids, DataType ids_data_type,
                           DataType table_ids_data_type, IDX* host_num_unique_matrix,
                           K* partitioned_unique_ids, U* partitioned_unique_table_ids,
                           K* received_ids, U* received_table_ids, int64_t* received_elem_cnt,
                           bool need_process_table_ids) {
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> send_elem_cnt;
  std::vector<int64_t> recv_offsets;
  std::vector<int64_t> recv_elem_cnt;
  MakeShuffleIdParams(host_num_unique_matrix, num_ids, 1, parallel_id, parallel_num, &send_offsets,
                      &send_elem_cnt, &recv_offsets, &recv_elem_cnt);
  ShuffleData(cuda_stream, comm, ids_data_type, send_offsets, send_elem_cnt, partitioned_unique_ids,
              recv_offsets, recv_elem_cnt, received_ids);
  *received_elem_cnt = recv_offsets.at(parallel_num - 1) + recv_elem_cnt.at(parallel_num - 1);
  if (need_process_table_ids) {
    ShuffleData(cuda_stream, comm, table_ids_data_type, send_offsets, send_elem_cnt,
                partitioned_unique_table_ids, recv_offsets, recv_elem_cnt, received_table_ids);
  }
}

template<typename K, typename IDX>
__global__ void UnsortedSegmentHalfGpu(const IDX in_h2_elem_cnt, const IDX h2_inner_dim_size,
                                       const IDX inner_dim_size, const half* data,
                                       const K* segment_ids, const IDX num_segments,
                                       half2* out_h2) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, in_h2_elem_cnt) {
    const IDX segment_id_idx = i / h2_inner_dim_size;
    const IDX h2_inner_idx = i - segment_id_idx * h2_inner_dim_size;
    const IDX inner_idx_0 = 2 * h2_inner_idx;
    const IDX inner_idx_1 = inner_idx_0 + 1;
    const half* data_row = data + segment_id_idx * inner_dim_size;
    half2 val;
    val.x = data_row[inner_idx_0];
    val.y = (inner_idx_1 >= inner_dim_size) ? static_cast<half>(0) : data_row[inner_idx_1];
    const IDX idx = segment_ids[segment_id_idx];
    const IDX out_h2_offset = idx * h2_inner_dim_size + h2_inner_idx;
    cuda::atomic::Add(out_h2 + out_h2_offset, val);
  }
}

template<typename T, typename K>
struct UnsortedSegmentSumPad {
  void operator()(ep::Stream* stream, const K* segment_ids, const T* data, int64_t num_segment_ids,
                  int64_t num_segments, int64_t inner_dim_size, int64_t padded_inner_dim_size,
                  T* out) const {
    UNIMPLEMENTED();
  }
};

template<typename K>
struct UnsortedSegmentSumPad<half, K> {
  void operator()(ep::Stream* stream, const K* segment_ids, const half* data,
                  int64_t num_segment_ids, int64_t num_segments, int64_t inner_dim_size,
                  int64_t padded_inner_dim_size, half* out) const {
    const int64_t data_elem_cnt = num_segment_ids * inner_dim_size;
    const int64_t out_elem_cnt = num_segments * padded_inner_dim_size;
    CHECK_EQ(padded_inner_dim_size % 2, 0);
    CHECK_EQ(inner_dim_size + 1, padded_inner_dim_size);
    const int64_t h2_inner_dim_size = padded_inner_dim_size / 2;
    const int64_t in_h2_elem_cnt = num_segment_ids * h2_inner_dim_size;
    if (std::max(data_elem_cnt, out_elem_cnt) < GetMaxVal<int32_t>() / 2) {
      UnsortedSegmentHalfGpu<K, int32_t>
          <<<BlocksNum4ThreadsNum(in_h2_elem_cnt), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(
              in_h2_elem_cnt, h2_inner_dim_size, inner_dim_size, data, segment_ids, num_segments,
              reinterpret_cast<half2*>(out));
    } else {
      UnsortedSegmentHalfGpu<K, int64_t>
          <<<BlocksNum4ThreadsNum(in_h2_elem_cnt), kCudaThreadsNumPerBlock, 0,
             stream->As<ep::CudaStream>()->cuda_stream()>>>(
              in_h2_elem_cnt, h2_inner_dim_size, inner_dim_size, data, segment_ids, num_segments,
              reinterpret_cast<half2*>(out));
    }
  }
};

template<typename T, typename K>
void UnsortedSegmentSum(ep::Stream* stream, const K* segment_ids, const T* data,
                        int64_t num_segment_ids, int64_t num_segments, int64_t inner_dim_size,
                        int64_t padded_inner_dim_size, T* out) {
  if (inner_dim_size == padded_inner_dim_size) {
    UnsortedSegmentSumKernelUtil<DeviceType::kCUDA, T, K, T>::UnsortedSegmentSum(
        stream, segment_ids, data, num_segment_ids, num_segments, 1, inner_dim_size, 0, out);
  } else {
    CHECK_EQ(inner_dim_size + 1, padded_inner_dim_size);
    UnsortedSegmentSumPad<T, K>()(stream, segment_ids, data, num_segment_ids, num_segments,
                                  inner_dim_size, padded_inner_dim_size, out);
  }
}

}  // namespace

template<typename K, typename V, typename IDX, typename HASH>
void UniqueAndPartition(cudaStream_t cuda_stream, int64_t num_ids, size_t capacity,
                        int64_t num_partition, const K* ids, const V* table_ids,
                        IDX* num_partitioned_unique_ids_ptr, K* partitioned_unique_ids,
                        V* partitioned_unique_table_ids, IDX* inverse_unique_partition_indices,
                        void* workspace_ptr, size_t workspace_bytes, bool need_process_table_ids,
                        const bool has_padding_idx, const int64_t padding_idx) {
  size_t table_capacity_bytes = capacity * sizeof(TableEntry<K>);
  CHECK_GE(workspace_bytes, table_capacity_bytes);
  OF_CUDA_CHECK(cudaMemsetAsync(workspace_ptr, 0, table_capacity_bytes, cuda_stream));
  OF_CUDA_CHECK(
      cudaMemsetAsync(num_partitioned_unique_ids_ptr, 0, num_partition * sizeof(IDX), cuda_stream));
  HashTableUniqueAndPartitionPairs<K, V, IDX, HASH>
      <<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
          capacity, num_ids, num_partition, num_partitioned_unique_ids_ptr,
          reinterpret_cast<TableEntry<K>*>(workspace_ptr), ids, table_ids, partitioned_unique_ids,
          partitioned_unique_table_ids, inverse_unique_partition_indices, need_process_table_ids,
          has_padding_idx, padding_idx);
}

template<typename T, typename IDX>
void ShuffleEmbeddings(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                       int64_t parallel_num, int64_t num_ids, int64_t embedding_size,
                       DataType data_type, IDX* host_num_unique_matrix,
                       const T* reverse_unique_cur_rank_embeddings, T* received_embeddings) {
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> send_elem_cnt;
  std::vector<int64_t> recv_offsets;
  std::vector<int64_t> recv_elem_cnt;
  MakeShuffleParams(host_num_unique_matrix, num_ids, embedding_size, parallel_id, parallel_num,
                    &recv_offsets, &recv_elem_cnt, &send_offsets, &send_elem_cnt);
  ShuffleData(cuda_stream, comm, data_type, send_offsets, send_elem_cnt,
              reverse_unique_cur_rank_embeddings, recv_offsets, recv_elem_cnt, received_embeddings);
}

// Quantized Version.
template<typename T, typename IDX>
void ShuffleEmbeddings(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                       int64_t parallel_num, int64_t num_ids, int64_t embedding_size,
                       DataType data_type, IDX* host_num_unique_matrix,
                       int8_t* reverse_unique_cur_rank_embeddings, int8_t* received_embeddings,
                       T* reverse_cur_rank_quantize_factor, T* recv_quantize_factor) {
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> send_elem_cnt;
  std::vector<int64_t> recv_offsets;
  std::vector<int64_t> recv_elem_cnt;
  // shuffle quantized_embedding
  MakeShuffleParams(host_num_unique_matrix, num_ids, embedding_size, parallel_id, parallel_num,
                    &recv_offsets, &recv_elem_cnt, &send_offsets, &send_elem_cnt);
  ShuffleData(cuda_stream, comm, DataType::kInt8, send_offsets, send_elem_cnt,
              reverse_unique_cur_rank_embeddings, recv_offsets, recv_elem_cnt, received_embeddings);
  // shuffle quantize_factor
  MakeShuffleParams(host_num_unique_matrix, num_ids, /*embedding_size=*/1, parallel_id,
                    parallel_num, &recv_offsets, &recv_elem_cnt, &send_offsets, &send_elem_cnt);
  ShuffleData(cuda_stream, comm, data_type, send_offsets, send_elem_cnt,
              reverse_cur_rank_quantize_factor, recv_offsets, recv_elem_cnt, recv_quantize_factor);
}

template<typename T, typename IDX>
void ShuffleEmbeddingsGrad(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                           int64_t parallel_num, int64_t num_ids, int64_t embedding_size,
                           DataType data_type, IDX* host_num_unique_matrix,
                           const T* unique_partition_embedding_grad, T* received_embeddings_grad) {
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> send_elem_cnt;
  std::vector<int64_t> recv_offsets;
  std::vector<int64_t> recv_elem_cnt;
  MakeShuffleParams(host_num_unique_matrix, num_ids, embedding_size, parallel_id, parallel_num,
                    &send_offsets, &send_elem_cnt, &recv_offsets, &recv_elem_cnt);
  ShuffleData(cuda_stream, comm, data_type, send_offsets, send_elem_cnt,
              unique_partition_embedding_grad, recv_offsets, recv_elem_cnt,
              received_embeddings_grad);
}

// Quantize Version.
template<typename T, typename IDX>
void ShuffleEmbeddingsGrad(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                           int64_t parallel_num, int64_t num_ids, int64_t embedding_size,
                           DataType data_type, IDX* host_num_unique_matrix,
                           int8_t* unique_partition_embedding_grad,
                           int8_t* received_embeddings_grad, T* cur_rank_quantize_factor,
                           T* received_cur_rank_quantize_factor) {
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> send_elem_cnt;
  std::vector<int64_t> recv_offsets;
  std::vector<int64_t> recv_elem_cnt;
  // Shuffle Embedding Grad.
  MakeShuffleParams(host_num_unique_matrix, num_ids, embedding_size, parallel_id, parallel_num,
                    &send_offsets, &send_elem_cnt, &recv_offsets, &recv_elem_cnt);
  ShuffleData(cuda_stream, comm, DataType::kInt8, send_offsets, send_elem_cnt,
              unique_partition_embedding_grad, recv_offsets, recv_elem_cnt,
              received_embeddings_grad);
  // Shuffle Quantize factor.
  MakeShuffleParams(host_num_unique_matrix, num_ids, /*embedding_size=*/1, parallel_id,
                    parallel_num, &send_offsets, &send_elem_cnt, &recv_offsets, &recv_elem_cnt);
  ShuffleData(cuda_stream, comm, data_type, send_offsets, send_elem_cnt, cur_rank_quantize_factor,
              recv_offsets, recv_elem_cnt, received_cur_rank_quantize_factor);
}

inline int64_t GetPaddedEmbeddingSize(DataType data_type, int64_t embedding_size) {
  if (data_type == DataType::kFloat16 && embedding_size % 2 != 0) {
    return embedding_size + 1;
  } else {
    return embedding_size;
  }
}

template<typename T, typename IDX>
void UniquePartitionEmbeddingGrad(ep::Stream* stream, int64_t unique_partitioned_num_ids,
                                  int64_t num_ids, int64_t embedding_size,
                                  int64_t padded_embedding_size, const IDX* host_num_unique_matrix,
                                  const T* embedding_grad,
                                  const IDX* inverse_unique_partition_indices,
                                  T* unique_partition_embedding_grad) {
  const int64_t valid_value_size = unique_partitioned_num_ids * padded_embedding_size * sizeof(T);
  OF_CUDA_CHECK(cudaMemsetAsync(unique_partition_embedding_grad, 0, valid_value_size,
                                stream->As<ep::CudaStream>()->cuda_stream()));
  UnsortedSegmentSum<T, IDX>(stream, inverse_unique_partition_indices, embedding_grad, num_ids,
                             unique_partitioned_num_ids, embedding_size, padded_embedding_size,
                             unique_partition_embedding_grad);
}

template<typename T, typename IDX>
void UniqueCurRankEmbeddingGrad(ep::Stream* stream, DataType data_type, int64_t cur_rank_num_ids,
                                int64_t num_unique, int64_t embedding_size,
                                int64_t padded_embedding_size, bool only_zero_valid_grad,
                                int64_t cur_rank_unique_embedding_grad_elem_cnt,
                                const T* cur_rank_embedding_grad,
                                const IDX* cur_rank_inverse_indices,
                                T* cur_rank_unique_embedding_grad, T* tmp_buffer) {
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  // memset cur_rank_unique_embedding_grad, if only_zero_valid_grad, only memset valid data.
  if (only_zero_valid_grad) {
    OF_CUDA_CHECK(cudaMemsetAsync(cur_rank_unique_embedding_grad, 0,
                                  num_unique * embedding_size * sizeof(T), cuda_stream));
  } else {
    OF_CUDA_CHECK(cudaMemsetAsync(cur_rank_unique_embedding_grad, 0,
                                  cur_rank_unique_embedding_grad_elem_cnt * sizeof(T),
                                  cuda_stream));
  }
  T* unsorted_segment_sum_out;
  if (embedding_size != padded_embedding_size) {
    unsorted_segment_sum_out = tmp_buffer;
    size_t buffer_size = GetCudaAlignedSize(num_unique * padded_embedding_size * sizeof(T));
    OF_CUDA_CHECK(cudaMemsetAsync(unsorted_segment_sum_out, 0, buffer_size, cuda_stream));
  } else {
    // cur_rank_unique_embedding_grad's has been memset, not need to memset again.
    unsorted_segment_sum_out = cur_rank_unique_embedding_grad;
  }
  UnsortedSegmentSum<T, IDX>(stream, cur_rank_inverse_indices, cur_rank_embedding_grad,
                             cur_rank_num_ids, num_unique, padded_embedding_size,
                             padded_embedding_size, unsorted_segment_sum_out);
  if (embedding_size != padded_embedding_size) {
    std::unique_ptr<ep::primitive::CopyNd> primitive =
        ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(DeviceType::kCUDA, 2);
    DimVector dst_shape = {num_unique, embedding_size};
    DimVector dst_pos_vec = {0, 0};
    DimVector src_shape = {num_unique, padded_embedding_size};
    DimVector src_pos_vec = {0, 0};
    DimVector extent_vec = {num_unique, embedding_size};
    primitive->Launch(stream, data_type, 2, cur_rank_unique_embedding_grad, dst_shape.data(),
                      dst_pos_vec.data(), unsorted_segment_sum_out, src_shape.data(),
                      src_pos_vec.data(), extent_vec.data());
  }
}

template<typename K, typename U, typename IDX>
struct IdShuffleDataPtrs {
  const K* ids_ptr;
  const U* table_ids_ptr;
  IDX* num_partitioned_unique;
  K* partitioned_unique_ids;
  U* partitioned_unique_table_ids;
  IDX* num_unique_matrix_ptr;
  IDX* inverse_unique_partition_indices_ptr;
  void* workspace_ptr;
  size_t workspace_size;
  K* received_ids;
  U* received_table_ids;
  IDX* cur_rank_num_unique_ptr;
  K* cur_rank_unique_ids_ptr;
  U* cur_rank_unique_table_ids_ptr;
  IDX* cur_rank_inverse_indices_ptr;
};

template<typename K, typename U, typename IDX>
void IdShuffle(ep::Stream* stream, ncclComm_t comm, const IdShuffleDataPtrs<K, U, IDX>& data_ptrs,
               int64_t num_ids, int64_t parallel_id, int64_t parallel_num,
               DataType num_unique_matrix_dtype, DataType ids_dtype, DataType table_ids_dtype,
               bool need_process_table_ids, const bool has_padding_idx, const int64_t padding_idx,
               IDX* host_num_unique_matrix, IDX* host_num_keys) {
  cudaStream_t cuda_stream = stream->As<ep::CudaStream>()->cuda_stream();
  size_t hash_table_capacity = parallel_num * num_ids;
  UniqueAndPartition<K, U, IDX, embedding::ShardingHash>(
      cuda_stream, num_ids, hash_table_capacity, parallel_num, data_ptrs.ids_ptr,
      data_ptrs.table_ids_ptr, data_ptrs.num_partitioned_unique, data_ptrs.partitioned_unique_ids,
      data_ptrs.partitioned_unique_table_ids, data_ptrs.inverse_unique_partition_indices_ptr,
      data_ptrs.workspace_ptr, data_ptrs.workspace_size, need_process_table_ids, has_padding_idx,
      padding_idx);

  OF_NCCL_CHECK(ncclAllGather(data_ptrs.num_partitioned_unique, data_ptrs.num_unique_matrix_ptr,
                              parallel_num, GetNcclDataType(num_unique_matrix_dtype), comm,
                              cuda_stream));

  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_unique_matrix, data_ptrs.num_unique_matrix_ptr,
                                parallel_num * parallel_num * sizeof(IDX), cudaMemcpyDefault,
                                cuda_stream));
  CHECK_JUST(stream->Sync());
  if (parallel_num > 1) {
    // use num_partitioned_unique as indices_offset buffer, so should after ncclAllGather.
    ComputeOffset<<<1, 1, 0, cuda_stream>>>(parallel_num, data_ptrs.num_partitioned_unique);
    ContiguousInverseUniquePartitionIndices<<<BlocksNum4ThreadsNum(num_ids),
                                              kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
        num_ids, data_ptrs.num_partitioned_unique, data_ptrs.inverse_unique_partition_indices_ptr);
  }
  int64_t received_elem_cnt = 0;
  ShuffleIdsAndTableIds(cuda_stream, comm, parallel_id, parallel_num, num_ids, ids_dtype,
                        table_ids_dtype, host_num_unique_matrix, data_ptrs.partitioned_unique_ids,
                        data_ptrs.partitioned_unique_table_ids, data_ptrs.received_ids,
                        data_ptrs.received_table_ids, &received_elem_cnt, need_process_table_ids);
  UniqueAndPartition<K, U, IDX, embedding::LocalUniqueHash>(
      cuda_stream, received_elem_cnt, hash_table_capacity, 1, data_ptrs.received_ids,
      data_ptrs.received_table_ids, data_ptrs.cur_rank_num_unique_ptr,
      data_ptrs.cur_rank_unique_ids_ptr, data_ptrs.cur_rank_unique_table_ids_ptr,
      data_ptrs.cur_rank_inverse_indices_ptr, data_ptrs.workspace_ptr, data_ptrs.workspace_size,
      need_process_table_ids, has_padding_idx, padding_idx);
  if (!need_process_table_ids) {
    OF_CUDA_CHECK(cudaMemsetAsync(data_ptrs.cur_rank_unique_table_ids_ptr, 0,
                                  received_elem_cnt * sizeof(U), cuda_stream));
  }
  OF_CUDA_CHECK(cudaMemcpyAsync(host_num_keys, data_ptrs.cur_rank_num_unique_ptr, sizeof(IDX),
                                cudaMemcpyDefault, cuda_stream));
  CHECK_JUST(stream->Sync());
}

}  // namespace data_shuffle
}  // namespace oneflow
