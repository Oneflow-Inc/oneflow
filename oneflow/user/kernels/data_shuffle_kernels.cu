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
#include <cub/cub.cuh>

namespace oneflow {

namespace {

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
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(void* ptr, const int64_t num_ids, const int64_t parallel_num) : ptr_(ptr) {
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
    const size_t received_num_unique_ids_bytes =
        GetCudaAlignedSize(parallel_num * parallel_num * sizeof(IDX));

    workspace_offset_ = 0;
    unique_ids_offset_ = workspace_offset_ + workspace_bytes_;
    partitioned_unique_ids_offset_ = unique_ids_offset_ + unique_ids_bytes;
    partitioned_num_unique_ids_offset_ =
        partitioned_unique_ids_offset_ + partitioned_unique_ids_bytes;
    received_unique_ids_offset_ =
        partitioned_num_unique_ids_offset_ + partitioned_num_unique_ids_bytes;
    received_num_unique_ids_offset_ = received_unique_ids_offset_ + received_unique_ids_bytes;
    CHECK_GE(workspace_bytes_, 0);
    total_buffer_size_ = workspace_bytes_ + unique_ids_bytes + partitioned_unique_ids_bytes
                         + partitioned_num_unique_ids_bytes + received_unique_ids_bytes
                         + received_num_unique_ids_bytes;
  }
  ~TmpBufferManager() = default;

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
  IDX* ReceivedNumUniqueIdsPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<IDX*>(reinterpret_cast<char*>(ptr_) + received_num_unique_ids_offset_);
  }

 private:
  size_t workspace_offset_;
  size_t unique_ids_offset_;
  size_t partitioned_unique_ids_offset_;
  size_t partitioned_num_unique_ids_offset_;
  size_t received_unique_ids_offset_;
  size_t received_num_unique_ids_offset_;

  size_t workspace_bytes_;
  size_t total_buffer_size_;
  void* ptr_;
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
    return std::make_shared<NcclKernelCommState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* nccl_comm = dynamic_cast<NcclKernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* ids = ctx->Tensor4ArgNameAndIndex("ids", 0);
    user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    user_op::Tensor* ids_reverse_idx = ctx->Tensor4ArgNameAndIndex("ids_reverse_idx", 0);
    user_op::Tensor* cur_rank_num_unique_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique_ids", 0);
    user_op::Tensor* cur_rank_unique_ids = ctx->Tensor4ArgNameAndIndex("cur_rank_unique_ids", 0);
    user_op::Tensor* cur_rank_reverse_idx = ctx->Tensor4ArgNameAndIndex("cur_rank_reverse_idx", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t num_ids = ids->shape().elem_cnt();
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    IDX* host_num_unique_ids;
    OF_CUDA_CHECK(
        cudaMallocHost(&host_num_unique_ids, (parallel_num * parallel_num + 1) * sizeof(IDX)));

    TmpBufferManager<K, IDX> buffer_manager(tmp_buffer->mut_dptr(), num_ids, parallel_num);
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
    IDX* received_num_unique_ids = buffer_manager.ReceivedNumUniqueIdsPtr();
    Partition(ctx->stream(), num_ids, host_num_unique_ids[0], parallel_num,
              buffer_manager.UniqueIdsPtr(), partitioned_unique_ids, partitioned_num_unique_ids,
              workspace_ptr, workspace_size);

    // allgather count
    ncclComm_t comm = nccl_comm->comm();
    OF_NCCL_CHECK(ncclAllGather(reinterpret_cast<const void*>(partitioned_num_unique_ids),
                                reinterpret_cast<void*>(received_num_unique_ids), parallel_num,
                                GetNcclDataType(cur_rank_num_unique_ids->data_type()), comm,
                                cuda_stream));
    IDX* host_received_num_unique_ids = host_num_unique_ids + 1;
    copyd2h_primitive->Launch(ctx->stream(), host_received_num_unique_ids, received_num_unique_ids,
                              parallel_num * parallel_num * sizeof(IDX));
    CHECK_JUST(ctx->stream()->Sync());

    // send recv
    int64_t recv_offset = 0;
    OF_NCCL_CHECK(ncclGroupStart());
    for (int64_t j = 0; j < parallel_num; ++j) {
      const int64_t need_send_elem_cnt =
          host_received_num_unique_ids[parallel_id * parallel_num + j];
      const int64_t need_recv_elem_cnt =
          host_received_num_unique_ids[j * parallel_num + parallel_id];
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
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename K, typename IDX>
user_op::InferTmpSizeFn GenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc& ids = ctx->InputTensorDesc("ids", 0);
    TmpBufferManager<K, IDX> buffer_manager(nullptr, ids.shape().elem_cnt(),
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
      .SetInferTmpSizeFn(GenInferTmpSizeFn<k_dtype, idx_dtype>());

// REGISTER_CUDA_ID_SHUFFLE_KERNEL(int32_t, int32_t)
REGISTER_CUDA_ID_SHUFFLE_KERNEL(int64_t, int32_t)

template<typename T>
class EmbeddingShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingShuffleKernel() = default;
  ~EmbeddingShuffleKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingShuffleKernel";
    const user_op::Tensor* cur_rank_embeddings =
        ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0);
    user_op::Tensor* cur_rank_num_unique_ids =
        ctx->Tensor4ArgNameAndIndex("cur_rank_num_unique_ids", 0);
    user_op::Tensor* ids_reverse_idx = ctx->Tensor4ArgNameAndIndex("cur_rank_reverse_idx", 0);
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(dtype)                  \
  REGISTER_USER_KERNEL("embedding_shuffle")                            \
      .SetCreateFn<EmbeddingShuffleKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("embeddings", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(float)

template<typename T>
class EmbeddingGradientShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingGradientShuffleKernel() = default;
  ~EmbeddingGradientShuffleKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingGradientShuffleKernel";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(dtype) \
  REGISTER_USER_KERNEL("embedding_gradient_shuffle")           \
      .SetCreateFn<EmbeddingGradientShuffleKernel<dtype>>()    \
      .SetIsMatchedHob(                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)      \
          && (user_op::HobDataType("embedding_diff", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(float)

}  // namespace oneflow
