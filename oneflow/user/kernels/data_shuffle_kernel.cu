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
#include "oneflow/core/ep/include/primitive/cast.h"

#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

namespace {

const size_t kUniquePartitionHashFuncSeed = 0;
const size_t kCurRankUniqueHashFuncSeed = 1;

template<typename K>
struct TableEntry;

template<>
struct TableEntry<int64_t> {
  int64_t key;
  uint32_t value;
};

template<>
struct TableEntry<int32_t> {
  int32_t key;
  uint32_t value;
};

template<>
struct TableEntry<uint64_t> {
  uint64_t key;
  uint32_t value;
};

template<>
struct TableEntry<uint32_t> {
  uint32_t key;
  uint32_t value;
};

template<typename K, typename V, typename IDX>
__global__ void HashTableUniqueAndPartitionPairs(const uint32_t table_capacity,
                                                 const uint32_t num_keys, int32_t num_partition,
                                                 size_t hash_func_seed, IDX* table_sizes,
                                                 TableEntry<K>* table, const K* keys,
                                                 const V* values, K* partioned_unique_keys,
                                                 V* partioned_unique_values, IDX* reverse_index,
                                                 bool need_process_values) {
  CUDA_1D_KERNEL_LOOP_T(uint32_t, i, num_keys) {
    IDX r_index_plus_one = 0;
    const K key = keys[i];
    size_t hash_key = XXH64()(key, hash_func_seed);
    uint32_t partition_id = hash_key % num_partition;
    IDX* table_size = table_sizes + partition_id;
    K* unique_keys = partioned_unique_keys + partition_id * num_keys;
    uint32_t pos = hash_key % table_capacity;
    const K key_hi = (key | 0x1);
    const K key_lo = (key & 0x1);
    uint32_t counter = 0;
    while (r_index_plus_one == 0) {
      bool prob_next = false;
      K* key_ptr = &table[pos].key;
      volatile uint32_t* value_ptr = &table[pos].value;
      const K old_key = cuda::atomic::CAS(key_ptr, 0, key_hi);
      if (old_key == 0) {
        IDX unique_pos = cuda::atomic::Add(table_size, 1);
        r_index_plus_one = unique_pos + 1;
        unique_keys[unique_pos] = key;
        if (need_process_values) {
          partioned_unique_values[partition_id * num_keys + unique_pos] = values[i];
        }
        *value_ptr = ((r_index_plus_one << 1U) | key_lo);
      } else if (old_key == key_hi) {
        const uint32_t value = *value_ptr;
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
    reverse_index[i] = r_index_plus_one - 1;
  }
}

template<typename U>
__global__ void GenerateColumnIds(int32_t elem_cnt, int32_t num_columns, U* column_ids) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { column_ids[i] = i % num_columns; }
}

template<typename K, typename V, typename IDX>
void UniqueAndPartition(cudaStream_t cuda_stream, int64_t num_ids, size_t capacity,
                        int64_t num_partition, size_t hash_func_seed, const K* ids,
                        const V* column_ids, IDX* num_partitioned_unique_ids_ptr,
                        K* partitioned_unique_ids, V* partitioned_unique_column_ids,
                        IDX* inverse_unique_partion_indices, void* workspace_ptr,
                        size_t workspace_bytes, bool need_process_column_ids) {
  CHECK_GE(workspace_bytes, capacity * sizeof(TableEntry<K>));
  OF_CUDA_CHECK(cudaMemsetAsync(workspace_ptr, 0, capacity * sizeof(TableEntry<K>), cuda_stream));
  OF_CUDA_CHECK(
      cudaMemsetAsync(num_partitioned_unique_ids_ptr, 0, num_partition * sizeof(IDX), cuda_stream));
  HashTableUniqueAndPartitionPairs<K, V, IDX>
      <<<BlocksNum4ThreadsNum(capacity), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
          capacity, num_ids, num_partition, hash_func_seed, num_partitioned_unique_ids_ptr,
          reinterpret_cast<TableEntry<K>*>(workspace_ptr), ids, column_ids, partitioned_unique_ids,
          partitioned_unique_column_ids, inverse_unique_partion_indices, need_process_column_ids);
}

template<typename K, typename U, typename IDX>
void SendRecvIdsAndColumns(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                           int64_t parallel_num, int64_t num_ids, DataType ids_data_type,
                           DataType column_ids_data_type, IDX* host_num_unique_matrix,
                           K* partitioned_unique_ids, U* partitioned_unique_column_ids,
                           K* received_ids, U* received_column_ids, int64_t* received_elem_cnt,
                           bool need_process_column_ids) {
  {
    // send recv unique ids
    int64_t recv_offset = 0;
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t j = 0; j < parallel_num; ++j) {
      const int64_t need_send_elem_cnt = host_num_unique_matrix[parallel_id * parallel_num + j];
      const int64_t need_recv_elem_cnt = host_num_unique_matrix[j * parallel_num + parallel_id];
      OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(partitioned_unique_ids + j * num_ids),
                             need_send_elem_cnt, GetNcclDataType(ids_data_type), j, comm,
                             cuda_stream));
      OF_NCCL_CHECK(ncclRecv(reinterpret_cast<void*>(received_ids + recv_offset),
                             need_recv_elem_cnt, GetNcclDataType(ids_data_type), j, comm,
                             cuda_stream));
      recv_offset += need_recv_elem_cnt;
    }
    OF_NCCL_CHECK(ncclGroupEnd());
    *received_elem_cnt = recv_offset;
  }
  {
    if (!need_process_column_ids) { return; }
    // send recv column_ids
    int64_t recv_offset = 0;
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t j = 0; j < parallel_num; ++j) {
      const int64_t need_send_elem_cnt = host_num_unique_matrix[parallel_id * parallel_num + j];
      const int64_t need_recv_elem_cnt = host_num_unique_matrix[j * parallel_num + parallel_id];
      OF_NCCL_CHECK(ncclSend(
          reinterpret_cast<const void*>(partitioned_unique_column_ids + j * num_ids),
          need_send_elem_cnt, GetNcclDataType(column_ids_data_type), j, comm, cuda_stream));
      OF_NCCL_CHECK(ncclRecv(reinterpret_cast<void*>(received_column_ids + recv_offset),
                             need_recv_elem_cnt, GetNcclDataType(column_ids_data_type), j, comm,
                             cuda_stream));
      recv_offset += need_recv_elem_cnt;
    }
    OF_NCCL_CHECK(ncclGroupEnd());
    CHECK_EQ(recv_offset, *received_elem_cnt);
  }
}

template<typename K, typename U, typename IDX>
class IdShuffleTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdShuffleTmpBufferManager);
  IdShuffleTmpBufferManager(void* ptr, const int64_t num_ids, const int64_t parallel_num,
                            bool need_column_ids, bool need_process_column_ids)
      : ptr_(ptr) {
    if (need_column_ids) { CHECK_EQ(need_process_column_ids, true); }
    if (!need_process_column_ids) { CHECK_EQ(need_column_ids, false); }
    const int64_t num_column_ids = need_process_column_ids ? num_ids : 0;
    const size_t column_ids_bytes = need_column_ids ? GetCudaAlignedSize(num_ids * sizeof(U)) : 0;
    const size_t num_partitioned_bytes = GetCudaAlignedSize(parallel_num * sizeof(IDX));
    const size_t partitioned_unique_ids_bytes =
        GetCudaAlignedSize(parallel_num * num_ids * sizeof(K));
    const size_t partitioned_unique_column_ids_bytes =
        GetCudaAlignedSize(parallel_num * num_column_ids * sizeof(U));
    const size_t received_ids_bytes = GetCudaAlignedSize(parallel_num * num_ids * sizeof(K));
    const size_t received_column_ids_bytes =
        GetCudaAlignedSize(parallel_num * num_column_ids * sizeof(U));
    const size_t hash_capacity = parallel_num * num_ids;
    workspace_bytes_ = GetCudaAlignedSize(hash_capacity * sizeof(TableEntry<K>));

    column_ids_offset_ = 0;
    num_partitioned_unique_offset_ = column_ids_offset_ + column_ids_bytes;
    partitioned_unique_ids_offset_ = num_partitioned_unique_offset_ + num_partitioned_bytes;
    partitioned_unique_column_ids_offset_ =
        partitioned_unique_ids_offset_ + partitioned_unique_ids_bytes;
    received_ids_offset_ =
        partitioned_unique_column_ids_offset_ + partitioned_unique_column_ids_bytes;
    received_column_ids_offset_ = received_ids_offset_ + received_ids_bytes;
    workspace_offset_ = received_column_ids_offset_ + received_column_ids_bytes;
    CHECK_GE(workspace_bytes_, 0);
    total_buffer_size_ = column_ids_bytes + num_partitioned_bytes + partitioned_unique_ids_bytes
                         + partitioned_unique_column_ids_bytes + received_ids_bytes
                         + received_column_ids_bytes + workspace_bytes_;
  }
  ~IdShuffleTmpBufferManager() = default;

  int64_t WorkspaceBytes() const { return workspace_bytes_; }
  size_t TotalBufferSize() const { return total_buffer_size_; }

  U* ColumnIds() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<U*>(reinterpret_cast<char*>(ptr_) + column_ids_offset_);
  }
  IDX* NumPartitionedUnique() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<IDX*>(reinterpret_cast<char*>(ptr_) + num_partitioned_unique_offset_);
  }
  K* PartitionedUniqueIds() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + partitioned_unique_ids_offset_);
  }
  U* PartitionedUniqueColumnIds() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<U*>(reinterpret_cast<char*>(ptr_)
                                + partitioned_unique_column_ids_offset_);
  }
  K* ReceivedIds() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + received_ids_offset_);
  }
  U* ReceivedColumnIds() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<U*>(reinterpret_cast<char*>(ptr_) + received_column_ids_offset_);
  }
  void* WorkspacePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr_) + workspace_offset_);
  }

 private:
  size_t column_ids_offset_;
  size_t num_partitioned_unique_offset_;
  size_t partitioned_unique_ids_offset_;
  size_t partitioned_unique_column_ids_offset_;
  size_t received_ids_offset_;
  size_t received_column_ids_offset_;
  size_t workspace_offset_;
  size_t workspace_bytes_;
  size_t total_buffer_size_;
  void* ptr_;
};

template<typename IDX>
class DataShuffleKernelState final : public user_op::OpKernelState {
 public:
  explicit DataShuffleKernelState(user_op::KernelInitContext* ctx)
      : has_independent_stream_(ctx->op_conf().has_stream_name_hint()),
        stream_name_(""),
        parallel_desc_(ctx->parallel_desc()) {
    if (has_independent_stream_) { stream_name_ = ctx->op_conf().stream_name_hint(); }
    OF_CUDA_CHECK(cudaMallocHost(
        &host_num_unique_matrix_,
        parallel_desc_.parallel_num() * parallel_desc_.parallel_num() * sizeof(IDX)));
  }
  ~DataShuffleKernelState() { OF_CUDA_CHECK(cudaFreeHost(host_num_unique_matrix_)); }

  ncclComm_t comm() { return GetOrCreate().comm; }

  IDX* HostNumUniqueMatrix() { return host_num_unique_matrix_; }

 private:
  struct Comm {
    Comm(ncclComm_t comm) : comm(comm) {}
    ncclComm_t comm;
  };

  const Comm& GetOrCreate() {
    if (!comm_) { Init(); }
    return *comm_;
  }

  void Init() {
    std::set<std::pair<int64_t, int64_t>> device_set;
    for (int64_t parallel_id = 0; parallel_id < parallel_desc_.parallel_num(); ++parallel_id) {
      int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
    ncclComm_t comm;
    if (has_independent_stream_) {
      comm = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
    } else {
      comm = comm_mgr->GetCommForDevice(device_set);
    }
    comm_.reset(new Comm(comm));
  }

  bool has_independent_stream_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  std::unique_ptr<Comm> comm_;
  IDX* host_num_unique_matrix_;
};

}  // namespace

template<typename K, typename U, typename IDX>
class IdShuffleKernel final : public user_op::OpKernel {
 public:
  IdShuffleKernel() = default;
  ~IdShuffleKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
    user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    user_op::Tensor* inverse_unique_partion_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partion_indices", 0);
    user_op::Tensor* cur_rank_num_unique = ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique", 0);
    user_op::Tensor* cur_rank_unique_ids = ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0);
    user_op::Tensor* cur_rank_unique_column_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_column_ids", 0);
    user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_columns = ctx->Attr<int32_t>("num_columns");
    const bool has_column_ids = ctx->has_input("column_ids", 0);
    const bool need_gen_column_ids = (!has_column_ids && num_columns > 1);
    const bool need_process_column_ids = (has_column_ids || num_columns > 1);
    const int64_t num_ids = ids->shape().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    IdShuffleTmpBufferManager<K, U, IDX> buffer_manager(tmp_buffer->mut_dptr(), num_ids,
                                                        parallel_num, need_gen_column_ids,
                                                        need_process_column_ids);
    CHECK_GE(tmp_buffer->shape().elem_cnt(), buffer_manager.TotalBufferSize());

    const U* column_ids_ptr;
    if (has_column_ids) {
      const user_op::Tensor* column_ids = ctx->Tensor4ArgNameAndIndex("column_ids", 0);
      column_ids_ptr = reinterpret_cast<const U*>(column_ids->dptr());
    } else if (need_gen_column_ids) {
      GenerateColumnIds<<<BlocksNum4ThreadsNum(num_ids), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
          num_ids, num_columns, buffer_manager.ColumnIds());
      column_ids_ptr = buffer_manager.ColumnIds();
    } else {
      column_ids_ptr = nullptr;
    }
    IDX* num_partitioned_unique = buffer_manager.NumPartitionedUnique();
    K* partitioned_unique_ids = buffer_manager.PartitionedUniqueIds();
    U* partitioned_unique_column_ids = buffer_manager.PartitionedUniqueColumnIds();
    IDX* num_unique_matrix_ptr = reinterpret_cast<IDX*>(num_unique_matrix->mut_dptr());
    size_t hash_capacity = parallel_num * num_ids;
    void* workspace_ptr = buffer_manager.WorkspacePtr();
    size_t workspace_size = buffer_manager.WorkspaceBytes();
    UniqueAndPartition<K, U, IDX>(
        cuda_stream, num_ids, hash_capacity, parallel_num, kUniquePartitionHashFuncSeed,
        reinterpret_cast<const K*>(ids->dptr()), column_ids_ptr, num_partitioned_unique,
        partitioned_unique_ids, partitioned_unique_column_ids,
        reinterpret_cast<IDX*>(inverse_unique_partion_indices->mut_dptr()), workspace_ptr,
        workspace_size, need_process_column_ids);
    ncclComm_t comm = kernel_state->comm();
    OF_NCCL_CHECK(ncclAllGather(reinterpret_cast<const void*>(num_partitioned_unique),
                                reinterpret_cast<void*>(num_unique_matrix_ptr), parallel_num,
                                GetNcclDataType(num_unique_matrix->data_type()), comm,
                                cuda_stream));
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_unique_matrix, num_unique_matrix_ptr,
                                  parallel_num * parallel_num * sizeof(IDX), cudaMemcpyDefault,
                                  cuda_stream));
    CHECK_JUST(ctx->stream()->Sync());

    K* received_ids = buffer_manager.ReceivedIds();
    U* received_column_ids = buffer_manager.ReceivedColumnIds();
    int64_t received_elem_cnt = 0;
    SendRecvIdsAndColumns(cuda_stream, comm, parallel_id, parallel_num, num_ids, ids->data_type(),
                          cur_rank_unique_column_ids->data_type(), host_num_unique_matrix,
                          partitioned_unique_ids, partitioned_unique_column_ids, received_ids,
                          received_column_ids, &received_elem_cnt, need_process_column_ids);
    UniqueAndPartition<K, U, IDX>(cuda_stream, received_elem_cnt, hash_capacity, 1,
                                  kCurRankUniqueHashFuncSeed, received_ids, received_column_ids,
                                  reinterpret_cast<IDX*>(cur_rank_num_unique->mut_dptr()),
                                  reinterpret_cast<K*>(cur_rank_unique_ids->mut_dptr()),
                                  reinterpret_cast<U*>(cur_rank_unique_column_ids->mut_dptr()),
                                  reinterpret_cast<IDX*>(cur_rank_inverse_indices->mut_dptr()),
                                  workspace_ptr, workspace_size, need_process_column_ids);
    if (!need_process_column_ids) {
      OF_CUDA_CHECK(cudaMemsetAsync(cur_rank_unique_column_ids->mut_dptr(), 0,
                                    cur_rank_unique_column_ids->shape().elem_cnt() * sizeof(U),
                                    cuda_stream));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define ID_DATA_TYPE_SEQ                            \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(uint64_t, DataType::kUInt64) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)   \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define IDX_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32)

#define REGISTER_CUDA_ID_SHUFFLE_KERNEL(k_dtype_pair, column_dtype_pair, idx_dtype_pair)          \
  REGISTER_USER_KERNEL("id_shuffle")                                                              \
      .SetCreateFn<                                                                               \
          IdShuffleKernel<OF_PP_PAIR_FIRST(k_dtype_pair), OF_PP_PAIR_FIRST(column_dtype_pair),    \
                          OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                                    \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("ids", 0) == OF_PP_PAIR_SECOND(k_dtype_pair))                  \
          && (user_op::HobDataType("cur_rank_unique_column_ids", 0)                               \
              == OF_PP_PAIR_SECOND(column_dtype_pair))                                            \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const user_op::TensorDesc& ids = ctx->InputTensorDesc("ids", 0);                          \
        const bool has_column_ids = ctx->has_input("column_ids", 0);                              \
        const int32_t num_columns = ctx->Attr<int32_t>("num_columns");                            \
        const bool need_gen_column_ids = (!has_column_ids && num_columns > 1);                    \
        const bool need_process_column_ids = (has_column_ids || num_columns > 1);                 \
        IdShuffleTmpBufferManager<OF_PP_PAIR_FIRST(k_dtype_pair),                                 \
                                  OF_PP_PAIR_FIRST(column_dtype_pair),                            \
                                  OF_PP_PAIR_FIRST(idx_dtype_pair)>                               \
            buffer_manager(nullptr, ids.shape().elem_cnt(), ctx->parallel_desc().parallel_num(),  \
                           need_gen_column_ids, need_process_column_ids);                         \
        return buffer_manager.TotalBufferSize();                                                  \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ID_SHUFFLE_KERNEL, ID_DATA_TYPE_SEQ,
                                 ID_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename IDX>
void SendRecvEmbeddings(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                        int64_t parallel_num, int64_t num_id, int64_t embedding_size,
                        DataType data_type, IDX* host_num_unique_matrix,
                        T* reverse_unique_cur_rank_embeddings, T* received_embeddings) {
  int64_t send_offset = 0;
  OF_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < parallel_num; ++i) {
    const int64_t need_send_elem_cnt =
        host_num_unique_matrix[i * parallel_num + parallel_id] * embedding_size;
    const int64_t need_recv_elem_cnt =
        host_num_unique_matrix[parallel_id * parallel_num + i] * embedding_size;
    OF_NCCL_CHECK(
        ncclSend(reinterpret_cast<const void*>(reverse_unique_cur_rank_embeddings + send_offset),
                 need_send_elem_cnt, GetNcclDataType(data_type), i, comm, cuda_stream));
    OF_NCCL_CHECK(
        ncclRecv(reinterpret_cast<void*>(received_embeddings + i * num_id * embedding_size),
                 need_recv_elem_cnt, GetNcclDataType(data_type), i, comm, cuda_stream));
    send_offset += need_send_elem_cnt;
  }
  OF_NCCL_CHECK(ncclGroupEnd());
}

template<typename T, typename IDX>
class EmbeddingShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingShuffleKernel() = default;
  ~EmbeddingShuffleKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* cur_rank_embeddings =
        ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0);
    const user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    const user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    const user_op::Tensor* inverse_unique_partion_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partion_indices", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t embedding_size = cur_rank_embeddings->shape().At(1);
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    DataType data_type = cur_rank_embeddings->data_type();
    const int64_t num_ids = inverse_unique_partion_indices->shape().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    OF_CUDA_CHECK(cudaMemcpyAsync(
        host_num_unique_matrix, reinterpret_cast<const IDX*>(num_unique_matrix->dptr()),
        parallel_num * parallel_num * sizeof(IDX), cudaMemcpyDefault, cuda_stream));
    CHECK_JUST(ctx->stream()->Sync());
    int64_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += host_num_unique_matrix[i * parallel_num + parallel_id];
    }

    size_t reverse_unique_cur_rank_embeddings_size =
        GetCudaAlignedSize(cur_rank_num_ids * embedding_size * sizeof(T));
    size_t received_embeddings_size =
        GetCudaAlignedSize(embeddings->shape().elem_cnt() * sizeof(T));
    T* reverse_unique_cur_rank_embeddings = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    T* received_embeddings = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                  + reverse_unique_cur_rank_embeddings_size);

    // reverse cur_rank unique
    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr()),
        cur_rank_num_ids, cur_rank_embeddings->dptr<T>(),
        Shape({1, cur_rank_embeddings->shape().elem_cnt() / embedding_size, embedding_size}),
        reverse_unique_cur_rank_embeddings, 0);

    ncclComm_t comm = kernel_state->comm();
    // send recv embeddings
    SendRecvEmbeddings(cuda_stream, comm, parallel_id, parallel_num, num_ids, embedding_size,
                       data_type, host_num_unique_matrix, reverse_unique_cur_rank_embeddings,
                       received_embeddings);

    // reverse unique_partition
    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), reinterpret_cast<const IDX*>(inverse_unique_partion_indices->dptr()),
        inverse_unique_partion_indices->shape().elem_cnt(), received_embeddings,
        Shape({1, parallel_num * num_ids, embedding_size}), embeddings->mut_dptr<T>(), 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(t_dtype_pair, idx_dtype_pair)                      \
  REGISTER_USER_KERNEL("embedding_shuffle")                                                       \
      .SetCreateFn<EmbeddingShuffleKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                         \
                                          OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                    \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("cur_rank_embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))  \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const user_op::TensorDesc& cur_rank_embeddings =                                          \
            ctx->InputTensorDesc("cur_rank_embeddings", 0);                                       \
        const user_op::TensorDesc& embeddings = ctx->InputTensorDesc("embeddings", 0);            \
        size_t reverse_cur_rank_embeddings_size = GetCudaAlignedSize(                             \
            cur_rank_embeddings.shape().elem_cnt() * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));     \
        size_t recv_unique_embeddings = GetCudaAlignedSize(                                       \
            embeddings.shape().elem_cnt() * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));              \
        return reverse_cur_rank_embeddings_size + recv_unique_embeddings;                         \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename IDX>
void SendRecvEmbeddingsDiff(cudaStream_t cuda_stream, ncclComm_t comm, int64_t parallel_id,
                            int64_t parallel_num, int64_t num_ids, int64_t embedding_size,
                            DataType data_type, IDX* host_num_unique_matrix,
                            T* unique_partition_embedding_diff, T* received_embeddings_diff) {
  int64_t recv_offset = 0;
  OF_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < parallel_num; ++i) {
    const int64_t need_send_elem_cnt =
        host_num_unique_matrix[parallel_id * parallel_num + i] * embedding_size;
    const int64_t need_recv_elem_cnt =
        host_num_unique_matrix[i * parallel_num + parallel_id] * embedding_size;
    OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(unique_partition_embedding_diff
                                                         + i * num_ids * embedding_size),
                           need_send_elem_cnt, GetNcclDataType(data_type), i, comm, cuda_stream));
    OF_NCCL_CHECK(ncclRecv(reinterpret_cast<void*>(received_embeddings_diff + recv_offset),
                           need_recv_elem_cnt, GetNcclDataType(data_type), i, comm, cuda_stream));
    recv_offset += need_recv_elem_cnt;
  }
  OF_NCCL_CHECK(ncclGroupEnd());
}

template<typename T, typename IDX>
class EmbeddingGradientShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingGradientShuffleKernel() = default;
  ~EmbeddingGradientShuffleKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    const user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    const user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    const user_op::Tensor* inverse_unique_partion_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partion_indices", 0);
    user_op::Tensor* cur_rank_unique_embedding_diff =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_embedding_diff", 0);
    const int64_t embedding_size = cur_rank_unique_embedding_diff->shape().At(1);
    IDX* host_num_unique_matrix = kernel_state->HostNumUniqueMatrix();
    DataType data_type = embedding_diff->data_type();
    const int64_t num_ids = inverse_unique_partion_indices->shape().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    OF_CUDA_CHECK(cudaMemcpyAsync(host_num_unique_matrix, num_unique_matrix->dptr(),
                                  parallel_num * parallel_num * sizeof(IDX), cudaMemcpyDefault,
                                  cuda_stream));
    CHECK_JUST(ctx->stream()->Sync());
    int64_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += host_num_unique_matrix[i * parallel_num + parallel_id];
    }

    size_t unique_partition_embedding_diff_size =
        GetCudaAlignedSize(parallel_num * num_ids * embedding_size * sizeof(T));
    size_t received_embedding_diff_size =
        GetCudaAlignedSize(parallel_num * num_ids * embedding_size * sizeof(T));
    T* unique_partition_embedding_diff = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    T* received_embedding_diff =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + unique_partition_embedding_diff_size);

    // unique and partion embedding diff
    OF_CUDA_CHECK(cudaMemsetAsync(unique_partition_embedding_diff, 0,
                                  parallel_num * num_ids * embedding_size, cuda_stream));
    UnsortedSegmentSumKernelUtil<DeviceType::kCUDA, T, IDX, T>::UnsortedSegmentSum(
        ctx->stream(), reinterpret_cast<const IDX*>(inverse_unique_partion_indices->dptr()),
        embedding_diff->dptr<T>(), num_ids, parallel_num * num_ids, 1, embedding_size, 0,
        unique_partition_embedding_diff);

    ncclComm_t comm = kernel_state->comm();
    // send recv embeddings diff
    SendRecvEmbeddingsDiff(cuda_stream, comm, parallel_id, parallel_num, num_ids, embedding_size,
                           data_type, host_num_unique_matrix, unique_partition_embedding_diff,
                           received_embedding_diff);

    // unique cur_rank embedding diff
    OF_CUDA_CHECK(cudaMemsetAsync(cur_rank_unique_embedding_diff->mut_dptr<T>(), 0,
                                  parallel_num * num_ids * embedding_size, cuda_stream));
    UnsortedSegmentSumKernelUtil<DeviceType::kCUDA, T, IDX, T>::UnsortedSegmentSum(
        ctx->stream(), reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr()),
        received_embedding_diff, cur_rank_num_ids, cur_rank_num_ids, 1, embedding_size, 0,
        cur_rank_unique_embedding_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(t_dtype_pair, idx_dtype_pair)             \
  REGISTER_USER_KERNEL("embedding_gradient_shuffle")                                              \
      .SetCreateFn<EmbeddingGradientShuffleKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                 \
                                                  OF_PP_PAIR_FIRST(idx_dtype_pair)>>()            \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("embedding_diff", 0) == OF_PP_PAIR_SECOND(t_dtype_pair))       \
          && (user_op::HobDataType("num_unique_matrix", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const user_op::TensorDesc& cur_rank_unique_embedding_diff =                               \
            ctx->InputTensorDesc("cur_rank_unique_embedding_diff", 0);                            \
        size_t cur_rank_embedding_diff_size =                                                     \
            GetCudaAlignedSize(cur_rank_unique_embedding_diff.shape().elem_cnt()                  \
                               * sizeof(OF_PP_PAIR_FIRST(t_dtype_pair)));                         \
        return 2 * cur_rank_embedding_diff_size;                                                  \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

}  // namespace oneflow
