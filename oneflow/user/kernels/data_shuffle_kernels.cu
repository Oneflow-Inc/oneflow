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
#include "oneflow/user/kernels/unique_kernel_util.h"
#include "oneflow/core/cuda/unique.cuh"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/gather_kernel_util.h"
#include "oneflow/user/kernels/unsorted_segment_sum_kernel_util.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

void DumpToFile(ep::Stream* stream, std::string filename, int64_t parallel_id, size_t data_size,
                const void* ptr) {
  void* host_ptr;
  OF_CUDA_CHECK(cudaMallocHost(&host_ptr, data_size));
  std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                ep::primitive::MemcpyKind::kDtoH);
  CHECK(copyd2h_primitive);
  copyd2h_primitive->Launch(stream, host_ptr, ptr, data_size);
  CHECK_JUST(stream->Sync());
  std::ofstream dx_os;
  dx_os.open(StrCat("test/" + filename + "_", parallel_id));
  dx_os.write(reinterpret_cast<char*>(host_ptr), data_size);
  dx_os.close();
  OF_CUDA_CHECK(cudaFreeHost(host_ptr));
}

template<typename T>
void DebugEmbeddingShuffle(user_op::KernelComputeContext* ctx) {
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  user_op::Tensor* cur_rank_embeddings = ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0);
  user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
  DumpToFile(ctx->stream(), "cur_rank_embeddings", parallel_id,
             cur_rank_embeddings->shape().elem_cnt() * sizeof(T), cur_rank_embeddings->dptr());
  DumpToFile(ctx->stream(), "embeddings", parallel_id, embeddings->shape().elem_cnt() * sizeof(T),
             embeddings->dptr());
}

template<typename T>
void DebugEmbeddingGradientShuffle(user_op::KernelComputeContext* ctx) {
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  user_op::Tensor* cur_rank_unique_embedding_diff =
      ctx->Tensor4ArgNameAndIndex("cur_rank_unique_embedding_diff", 0);
  user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
  DumpToFile(ctx->stream(), "cur_rank_unique_embedding_diff", parallel_id,
             cur_rank_unique_embedding_diff->shape().elem_cnt() * sizeof(T),
             cur_rank_unique_embedding_diff->dptr());
  DumpToFile(ctx->stream(), "embedding_diff", parallel_id,
             embedding_diff->shape().elem_cnt() * sizeof(T), embedding_diff->dptr());
}

template<typename K, typename IDX>
void DebugIdShuffle(user_op::KernelComputeContext* ctx) {
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
  user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
  user_op::Tensor* ids_reverse_idx = ctx->Tensor4ArgNameAndIndex("ids_reverse_idx", 0);
  user_op::Tensor* cur_rank_num_unique_ids =
      ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique_ids", 0);
  user_op::Tensor* cur_rank_unique_ids = ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0);
  user_op::Tensor* cur_rank_reverse_idx = ctx->Tensor4ArgNameAndIndex("cur_rank_reverse_idx", 0);
  user_op::Tensor* num_unique_ids_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_ids_matrix", 0);
  DumpToFile(ctx->stream(), "ids", parallel_id, ids->shape().elem_cnt() * sizeof(K), ids->dptr());
  DumpToFile(ctx->stream(), "num_unique_ids", parallel_id,
             num_unique_ids->shape().elem_cnt() * sizeof(IDX), num_unique_ids->dptr());
  DumpToFile(ctx->stream(), "ids_reverse_idx", parallel_id,
             ids_reverse_idx->shape().elem_cnt() * sizeof(IDX), ids_reverse_idx->dptr());
  DumpToFile(ctx->stream(), "cur_rank_num_unique_ids", parallel_id,
             cur_rank_num_unique_ids->shape().elem_cnt() * sizeof(IDX),
             cur_rank_num_unique_ids->dptr());
  DumpToFile(ctx->stream(), "cur_rank_unique_ids", parallel_id,
             cur_rank_unique_ids->shape().elem_cnt() * sizeof(K), cur_rank_unique_ids->dptr());
  DumpToFile(ctx->stream(), "cur_rank_reverse_idx", parallel_id,
             cur_rank_reverse_idx->shape().elem_cnt() * sizeof(IDX), cur_rank_reverse_idx->dptr());
  DumpToFile(ctx->stream(), "num_unique_ids_matrix", parallel_id,
             num_unique_ids_matrix->shape().elem_cnt() * sizeof(IDX),
             num_unique_ids_matrix->dptr());
}

struct BelongTo {
  int parallel_num;
  int parallel_id;
  __host__ __device__ __forceinline__ BelongTo(int parallel_num, int parallel_id)
      : parallel_num(parallel_num), parallel_id(parallel_id) {}
  __host__ __device__ __forceinline__ bool operator()(const int& a) const {
    return (a % parallel_num) == parallel_id;
  }
};

template<typename K, typename IDX>
void GetPartitionWorkspaceSizeInBytes(ep::Stream* stream, int64_t n, int64_t parallel_num,
                                      size_t* workspace_size_in_bytes) {
  BelongTo belong_to(parallel_num, 0);
  cub::DeviceSelect::If<K*, K*, IDX*>(nullptr, *workspace_size_in_bytes, nullptr, nullptr, nullptr,
                                      n, belong_to, stream->As<ep::CudaStream>()->cuda_stream());
}

template<typename K, typename IDX>
void Partition(ep::Stream* stream, int64_t num_ids, IDX num_valid, int64_t parallel_num, K* in,
               K* out, IDX* num_out, void* workspace, size_t workspace_size_in_bytes) {
  for (int64_t i = 0; i < parallel_num; ++i) {
    BelongTo belong_to(parallel_num, i);
    cub::DeviceSelect::If(workspace, workspace_size_in_bytes, in, out + i * num_ids, num_out + i,
                          num_valid, belong_to, stream->As<ep::CudaStream>()->cuda_stream());
  }
}

template<typename K, typename IDX>
class IdShuffleTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdShuffleTmpBufferManager);
  IdShuffleTmpBufferManager(void* ptr, const int64_t num_ids, const int64_t parallel_num)
      : ptr_(ptr) {
    int64_t unique_workspace_bytes = 0;
    UniqueKernelUtil<DeviceType::kCUDA, K, IDX>::GetUniqueWorkspaceSizeInBytes(
        nullptr, parallel_num * num_ids, &unique_workspace_bytes);
    // size_t partition_workspace_bytes = 0;
    // TODO: GetPartitionWorkspaceSizeInBytes have bug?
    // GetPartitionWorkspaceSizeInBytes<K, IDX>(nullptr, num_ids, parallel_num,
    //                                         &partition_workspace_bytes);
    // workspace_bytes_ = GetCudaAlignedSize(
    //    std::max(static_cast<size_t>(unique_workspace_bytes), partition_workspace_bytes));
    workspace_bytes_ = GetCudaAlignedSize(unique_workspace_bytes);
    const size_t unique_ids_bytes = GetCudaAlignedSize(num_ids * sizeof(K));
    const size_t partitioned_unique_ids_bytes =
        GetCudaAlignedSize(parallel_num * num_ids * sizeof(K));
    const size_t partitioned_num_unique_ids_bytes = GetCudaAlignedSize(parallel_num * sizeof(IDX));
    const size_t received_unique_ids_bytes = GetCudaAlignedSize(parallel_num * num_ids * sizeof(K));

    workspace_offset_ = 0;
    unique_ids_offset_ = workspace_offset_ + workspace_bytes_;
    partitioned_unique_ids_offset_ = unique_ids_offset_ + unique_ids_bytes;
    partitioned_num_unique_ids_offset_ =
        partitioned_unique_ids_offset_ + partitioned_unique_ids_bytes;
    received_unique_ids_offset_ =
        partitioned_num_unique_ids_offset_ + partitioned_num_unique_ids_bytes;
    CHECK_GE(workspace_bytes_, 0);
    total_buffer_size_ = workspace_bytes_ + unique_ids_bytes + partitioned_unique_ids_bytes
                         + partitioned_num_unique_ids_bytes + received_unique_ids_bytes;
  }
  ~IdShuffleTmpBufferManager() = default;

  int64_t WorkspaceBytes() const { return workspace_bytes_; }
  size_t TotalBufferSize() const { return total_buffer_size_; }

  void* WorkspacePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr_) + workspace_offset_);
  }
  K* UniqueIdsPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + unique_ids_offset_);
  }
  K* PartitionedUniqueIdsPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + partitioned_unique_ids_offset_);
  }
  IDX* PartitionedNumUniqueIdsPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<IDX*>(reinterpret_cast<char*>(ptr_)
                                  + partitioned_num_unique_ids_offset_);
  }
  K* ReceivedUniqueIdsPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + received_unique_ids_offset_);
  }

 private:
  size_t workspace_offset_;
  size_t unique_ids_offset_;
  size_t partitioned_unique_ids_offset_;
  size_t partitioned_num_unique_ids_offset_;
  size_t received_unique_ids_offset_;

  size_t workspace_bytes_;
  size_t total_buffer_size_;
  void* ptr_;
};

class IdShuffleNcclKernelCommState final : public user_op::OpKernelState {
 public:
  explicit IdShuffleNcclKernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false), parallel_desc_(ctx->parallel_desc()) {}
  ~IdShuffleNcclKernelCommState() = default;

  ncclComm_t comm() {
    if (!is_init_) {
      std::set<std::pair<int64_t, int64_t>> device_set;
      FOR_RANGE(int64_t, parallel_id, 0, parallel_desc_.parallel_num()) {
        int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
        int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
        device_set.emplace(std::make_pair(machine_id, device_id));
      }
      EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
      comm_ = comm_mgr->GetCommForDeviceAndStreamName(device_set, "ID_SHUFFLE");
      is_init_ = true;
    }
    return comm_;
  }

 private:
  bool is_init_;
  ParallelDesc parallel_desc_;
  ncclComm_t comm_{};
};

class NcclKernelCommState final : public user_op::OpKernelState {
 public:
  explicit NcclKernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false), parallel_desc_(ctx->parallel_desc()) {}
  ~NcclKernelCommState() = default;

  ncclComm_t comm() {
    if (!is_init_) {
      std::set<std::pair<int64_t, int64_t>> device_set;
      FOR_RANGE(int64_t, parallel_id, 0, parallel_desc_.parallel_num()) {
        int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
        int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
        device_set.emplace(std::make_pair(machine_id, device_id));
      }
      EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
      comm_ = comm_mgr->GetCommForDevice(device_set);
      is_init_ = true;
    }
    return comm_;
  }

 private:
  bool is_init_;
  ParallelDesc parallel_desc_;
  ncclComm_t comm_{};
};

}  // namespace

template<typename K, typename IDX>
class IdShuffleKernel final : public user_op::OpKernel {
 public:
  IdShuffleKernel() = default;
  ~IdShuffleKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<IdShuffleNcclKernelCommState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* nccl_comm = dynamic_cast<IdShuffleNcclKernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
    user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    user_op::Tensor* ids_reverse_idx = ctx->Tensor4ArgNameAndIndex("ids_reverse_idx", 0);
    user_op::Tensor* cur_rank_num_unique_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique_ids", 0);
    user_op::Tensor* cur_rank_unique_ids = ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0);
    user_op::Tensor* cur_rank_reverse_idx = ctx->Tensor4ArgNameAndIndex("cur_rank_reverse_idx", 0);
    user_op::Tensor* num_unique_ids_matrix =
        ctx->Tensor4ArgNameAndIndex("num_unique_ids_matrix", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t num_ids = ids->shape().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    IDX* host_num_unique_ids;
    OF_CUDA_CHECK(
        cudaMallocHost(&host_num_unique_ids, (parallel_num * parallel_num + 1) * sizeof(IDX)));

    IdShuffleTmpBufferManager<K, IDX> buffer_manager(tmp_buffer->mut_dptr(), num_ids, parallel_num);
    void* workspace_ptr = buffer_manager.WorkspacePtr();
    size_t workspace_size = buffer_manager.WorkspaceBytes();
    // unique
    UniqueKernelUtil<DeviceType::kCUDA, K, IDX>::Unique(
        ctx->stream(), num_ids, ids->dptr<K>(), num_unique_ids->mut_dptr<IDX>(),
        buffer_manager.UniqueIdsPtr(), ids_reverse_idx->mut_dptr<IDX>(), workspace_ptr,
        workspace_size);
    // partition
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_unique_ids, num_unique_ids->mut_dptr(),
                              sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());

    LOG(ERROR) << "rank " << parallel_id << " num_unique_ids " << *host_num_unique_ids;
    K* partitioned_unique_ids = buffer_manager.PartitionedUniqueIdsPtr();
    IDX* partitioned_num_unique_ids = buffer_manager.PartitionedNumUniqueIdsPtr();
    K* received_unique_ids = buffer_manager.ReceivedUniqueIdsPtr();
    /*
    num_unique_ids_matrix(parallel_num * parallel_num):
           partion0   partion1
    rank0
    rank1
    */
    IDX* received_num_unique_ids_matrix = num_unique_ids_matrix->mut_dptr<IDX>();
    Partition(ctx->stream(), num_ids, host_num_unique_ids[0], parallel_num,
              buffer_manager.UniqueIdsPtr(), partitioned_unique_ids, partitioned_num_unique_ids,
              workspace_ptr, workspace_size);

    // allgather count
    ncclComm_t comm = nccl_comm->comm();
    OF_NCCL_CHECK(ncclAllGather(reinterpret_cast<const void*>(partitioned_num_unique_ids),
                                reinterpret_cast<void*>(received_num_unique_ids_matrix),
                                parallel_num, GetNcclDataType(cur_rank_num_unique_ids->data_type()),
                                comm, cuda_stream));
    IDX* host_num_unique_ids_matrix = host_num_unique_ids + 1;
    copyd2h_primitive->Launch(ctx->stream(), host_num_unique_ids_matrix,
                              received_num_unique_ids_matrix,
                              parallel_num * parallel_num * sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());

    // send recv
    int64_t recv_offset = 0;
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t j = 0; j < parallel_num; ++j) {
      const int64_t need_send_elem_cnt = host_num_unique_ids_matrix[parallel_id * parallel_num + j];
      const int64_t need_recv_elem_cnt = host_num_unique_ids_matrix[j * parallel_num + parallel_id];
      OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(partitioned_unique_ids + j * num_ids),
                             need_send_elem_cnt, GetNcclDataType(ids->data_type()), j, comm,
                             cuda_stream));
      OF_NCCL_CHECK(ncclRecv(reinterpret_cast<void*>(received_unique_ids + recv_offset),
                             need_recv_elem_cnt, GetNcclDataType(ids->data_type()), j, comm,
                             cuda_stream));
      recv_offset += need_recv_elem_cnt;
    }
    OF_NCCL_CHECK(ncclGroupEnd());
    // unique
    UniqueKernelUtil<DeviceType::kCUDA, K, IDX>::Unique(
        ctx->stream(), recv_offset, received_unique_ids, cur_rank_num_unique_ids->mut_dptr<IDX>(),
        cur_rank_unique_ids->mut_dptr<K>(), cur_rank_reverse_idx->mut_dptr<IDX>(), workspace_ptr,
        workspace_size);

    if (ParseBooleanFromEnv("DEBUG_SHUFFLE", false)) { DebugIdShuffle<K, IDX>(ctx); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename K, typename IDX>
user_op::InferTmpSizeFn GenIdShuffleInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& ids = ctx->InputTensorDesc("ids", 0);
    IdShuffleTmpBufferManager<K, IDX> buffer_manager(nullptr, ids.shape().elem_cnt(),
                                                     ctx->parallel_desc().parallel_num());
    return buffer_manager.TotalBufferSize();
  };
}

#define REGISTER_CUDA_ID_SHUFFLE_KERNEL(k_dtype, idx_dtype)                                \
  REGISTER_USER_KERNEL("id_shuffle")                                                       \
      .SetCreateFn<IdShuffleKernel<k_dtype, idx_dtype>>()                                  \
      .SetIsMatchedHob(                                                                    \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                  \
          && (user_op::HobDataType("ids", 0) == GetDataType<k_dtype>::value)               \
          && (user_op::HobDataType("num_unique_ids", 0) == GetDataType<idx_dtype>::value)) \
      .SetInferTmpSizeFn(GenIdShuffleInferTmpSizeFn<k_dtype, idx_dtype>());

// REGISTER_CUDA_ID_SHUFFLE_KERNEL(int32_t, int32_t)
REGISTER_CUDA_ID_SHUFFLE_KERNEL(int64_t, int32_t)

template<typename T, typename IDX>
class EmbeddingShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingShuffleKernel() = default;
  ~EmbeddingShuffleKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclKernelCommState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* nccl_comm = dynamic_cast<NcclKernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    ncclComm_t comm = nccl_comm->comm();

    LOG(ERROR) << "EmbeddingShuffleKernel";
    const user_op::Tensor* cur_rank_embeddings =
        ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0);
    const user_op::Tensor* cur_rank_num_unique_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique_ids", 0);
    const user_op::Tensor* cur_rank_reverse_idx =
        ctx->Tensor4ArgNameAndIndex("cur_rank_reverse_idx", 0);
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* ids_reverse_idx = ctx->Tensor4ArgNameAndIndex("ids_reverse_idx", 0);
    const user_op::Tensor* num_unique_ids_matrix =
        ctx->Tensor4ArgNameAndIndex("num_unique_ids_matrix", 0);
    user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t num_ids = ids_reverse_idx->shape().elem_cnt();
    const int64_t embedding_size = embeddings->shape().elem_cnt() / num_ids;
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();

    IDX* host_num_unique_ids_matrix;
    OF_CUDA_CHECK(
        cudaMallocHost(&host_num_unique_ids_matrix, parallel_num * parallel_num * sizeof(IDX)));
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_unique_ids_matrix,
                              num_unique_ids_matrix->dptr(),
                              parallel_num * parallel_num * sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());
    int64_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += host_num_unique_ids_matrix[i * parallel_num + parallel_id];
    }

    LOG(ERROR) << "parallel_id " << parallel_id << " cur_rank_num_ids " << cur_rank_num_ids;
    size_t reverse_cur_rank_embeddings_size =
        GetCudaAlignedSize(cur_rank_num_ids * embedding_size * sizeof(T));
    T* reverse_cur_rank_embeddings = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    T* recv_unique_embeddings =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + reverse_cur_rank_embeddings_size);

    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), cur_rank_reverse_idx->dptr<IDX>(), cur_rank_num_ids,
        cur_rank_embeddings->dptr<T>(),
        Shape({1, cur_rank_embeddings->shape().elem_cnt() / embedding_size, embedding_size}),
        reverse_cur_rank_embeddings, 0);
    int64_t send_offset = 0;
    int64_t recv_offset = 0;
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t i = 0; i < parallel_num; ++i) {
      const int64_t need_send_elem_cnt =
          host_num_unique_ids_matrix[i * parallel_num + parallel_id] * embedding_size;
      const int64_t need_recv_elem_cnt =
          host_num_unique_ids_matrix[parallel_id * parallel_num + i] * embedding_size;
      LOG(ERROR) << "need_send_elem_cnt "
                 << host_num_unique_ids_matrix[i * parallel_num + parallel_id];
      LOG(ERROR) << " need_recv_elem_cnt "
                 << host_num_unique_ids_matrix[parallel_id * parallel_num + i];
      OF_NCCL_CHECK(
          ncclSend(reinterpret_cast<const void*>(reverse_cur_rank_embeddings + send_offset),
                   need_send_elem_cnt, GetNcclDataType(cur_rank_embeddings->data_type()), i, comm,
                   cuda_stream));
      OF_NCCL_CHECK(ncclRecv(reinterpret_cast<void*>(recv_unique_embeddings + recv_offset),
                             need_recv_elem_cnt, GetNcclDataType(cur_rank_embeddings->data_type()),
                             i, comm, cuda_stream));
      send_offset += need_send_elem_cnt;
      recv_offset += need_recv_elem_cnt;
    }
    OF_NCCL_CHECK(ncclGroupEnd());
    GatherKernelUtilImpl<DeviceType::kCUDA, T, IDX>::Forward(
        ctx->stream(), ids_reverse_idx->dptr<IDX>(), ids_reverse_idx->shape().elem_cnt(),
        recv_unique_embeddings, Shape({1, num_ids, embedding_size}), embeddings->mut_dptr<T>(), 0);

    if (ParseBooleanFromEnv("DEBUG_SHUFFLE", false)) { DebugEmbeddingShuffle<T>(ctx); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename IDX>
user_op::InferTmpSizeFn GenEmbeddingShuffleInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& cur_rank_embeddings = ctx->InputTensorDesc("cur_rank_embeddings", 0);
    const user_op::TensorDesc& embeddings = ctx->InputTensorDesc("embeddings", 0);
    size_t reverse_cur_rank_embeddings_size =
        GetCudaAlignedSize(cur_rank_embeddings.shape().elem_cnt() * sizeof(T));
    size_t recv_unique_embeddings = GetCudaAlignedSize(embeddings.shape().elem_cnt() * sizeof(T));
    return reverse_cur_rank_embeddings_size + recv_unique_embeddings;
  };
}

#define REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(t_dtype, idx_dtype)                                 \
  REGISTER_USER_KERNEL("embedding_shuffle")                                                        \
      .SetCreateFn<EmbeddingShuffleKernel<t_dtype, idx_dtype>>()                                   \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                          \
          && (user_op::HobDataType("cur_rank_num_unique_ids", 0) == GetDataType<idx_dtype>::value) \
          && (user_op::HobDataType("embeddings", 0) == GetDataType<t_dtype>::value))               \
      .SetInferTmpSizeFn(GenEmbeddingShuffleInferTmpSizeFn<t_dtype, idx_dtype>());

REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(float, int32_t)

template<typename T, typename IDX>
class EmbeddingGradientShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingGradientShuffleKernel() = default;
  ~EmbeddingGradientShuffleKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclKernelCommState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* nccl_comm = dynamic_cast<NcclKernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    ncclComm_t comm = nccl_comm->comm();
    LOG(ERROR) << "EmbeddingGradientShuffleKernel";

    const user_op::Tensor* embedding_diff = ctx->Tensor4ArgNameAndIndex("embedding_diff", 0);
    const user_op::Tensor* cur_rank_num_unique_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique_ids", 0);
    const user_op::Tensor* cur_rank_reverse_idx =
        ctx->Tensor4ArgNameAndIndex("cur_rank_reverse_idx", 0);
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* ids_reverse_idx = ctx->Tensor4ArgNameAndIndex("ids_reverse_idx", 0);
    const user_op::Tensor* num_unique_ids_matrix =
        ctx->Tensor4ArgNameAndIndex("num_unique_ids_matrix", 0);
    user_op::Tensor* cur_rank_unique_embedding_diff =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_embedding_diff", 0);
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t num_ids = ids_reverse_idx->shape().elem_cnt();
    const int64_t embedding_size = embedding_diff->shape().elem_cnt() / num_ids;
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    IDX* host_num_unique_ids_matrix;
    OF_CUDA_CHECK(
        cudaMallocHost(&host_num_unique_ids_matrix, parallel_num * parallel_num * sizeof(IDX)));
    std::unique_ptr<ep::primitive::Memcpy> copyd2h_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(DeviceType::kCUDA,
                                                                  ep::primitive::MemcpyKind::kDtoH);
    CHECK(copyd2h_primitive);
    copyd2h_primitive->Launch(ctx->stream(), host_num_unique_ids_matrix,
                              num_unique_ids_matrix->dptr(),
                              parallel_num * parallel_num * sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());
    int64_t cur_rank_num_ids = 0;
    for (int64_t i = 0; i < parallel_num; ++i) {
      cur_rank_num_ids += host_num_unique_ids_matrix[i * parallel_num + parallel_id];
    }

    size_t unique_diff_size = embedding_diff->shape().elem_cnt() * sizeof(T);
    T* unique_diff_ptr = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    T* recv_embeddings_diff = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + unique_diff_size);

    Memset<DeviceType::kCUDA>(ctx->stream(), unique_diff_ptr, 0,
                              embedding_diff->shape().elem_cnt() * sizeof(T));
    UnsortedSegmentSumKernelUtil<DeviceType::kCUDA, T, IDX, T>::UnsortedSegmentSum(
        ctx->stream(), ids_reverse_idx->dptr<IDX>(), embedding_diff->dptr<T>(), num_ids, num_ids, 1,
        embedding_size, 0, unique_diff_ptr);

    int64_t send_offset = 0;
    int64_t recv_offset = 0;
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t i = 0; i < parallel_num; ++i) {
      const int64_t need_send_elem_cnt =
          host_num_unique_ids_matrix[parallel_id * parallel_num + i] * embedding_size;
      const int64_t need_recv_elem_cnt =
          host_num_unique_ids_matrix[i * parallel_num + parallel_id] * embedding_size;
      OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(unique_diff_ptr + send_offset),
                             need_send_elem_cnt, GetNcclDataType(embedding_diff->data_type()), i,
                             comm, cuda_stream));
      OF_NCCL_CHECK(ncclRecv(reinterpret_cast<void*>(recv_embeddings_diff + recv_offset),
                             need_recv_elem_cnt, GetNcclDataType(embedding_diff->data_type()), i,
                             comm, cuda_stream));
      send_offset += need_send_elem_cnt;
      recv_offset += need_recv_elem_cnt;
    }
    OF_NCCL_CHECK(ncclGroupEnd());

    Memset<DeviceType::kCUDA>(ctx->stream(), cur_rank_unique_embedding_diff->mut_dptr<T>(), 0,
                              cur_rank_unique_embedding_diff->shape().elem_cnt() * sizeof(T));
    UnsortedSegmentSumKernelUtil<DeviceType::kCUDA, T, IDX, T>::UnsortedSegmentSum(
        ctx->stream(), cur_rank_reverse_idx->dptr<IDX>(), recv_embeddings_diff, cur_rank_num_ids,
        cur_rank_num_ids, 1, embedding_size, 0, cur_rank_unique_embedding_diff->mut_dptr<T>());

    if (ParseBooleanFromEnv("DEBUG_SHUFFLE", false)) { DebugEmbeddingGradientShuffle<T>(ctx); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename IDX>
user_op::InferTmpSizeFn GenEmbeddingGradientShuffleInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& embedding_diff = ctx->InputTensorDesc("embedding_diff", 0);
    const user_op::TensorDesc& cur_rank_unique_embedding_diff =
        ctx->InputTensorDesc("cur_rank_unique_embedding_diff", 0);
    size_t unique_embedding_diff_size = embedding_diff.shape().elem_cnt() * sizeof(T);
    size_t cur_rank_unique_embedding_diff_size =
        cur_rank_unique_embedding_diff.shape().elem_cnt() * sizeof(T);
    return unique_embedding_diff_size + cur_rank_unique_embedding_diff_size;
  };
}

#define REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(t_dtype, idx_dtype)                \
  REGISTER_USER_KERNEL("embedding_gradient_shuffle")                                       \
      .SetCreateFn<EmbeddingGradientShuffleKernel<t_dtype, idx_dtype>>()                   \
      .SetIsMatchedHob(                                                                    \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                  \
          && (user_op::HobDataType("ids_reverse_idx", 0) == GetDataType<idx_dtype>::value) \
          && (user_op::HobDataType("embedding_diff", 0) == GetDataType<t_dtype>::value))   \
      .SetInferTmpSizeFn(GenEmbeddingGradientShuffleInferTmpSizeFn<t_dtype, idx_dtype>());

REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(float, int32_t)

}  // namespace oneflow
