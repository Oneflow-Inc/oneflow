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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include <cuda.h>

#if CUDA_VERSION >= 11030

namespace oneflow {

namespace {

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename T, typename IDX, int pack_size, int N>
struct Param {
  IDX* inverse_indices[N];
  Pack<T, pack_size>* unique_embeddings[N];
  int32_t* is_kernel_start[N];
  const IDX* num_unique_matrix;
  Pack<T, pack_size>* embedding_ptr;
};

template<typename T, typename IDX, int pack_size, int N>
__global__ void EmbeddingShuffleCudaKernel(int parallel_id, int parallel_num,
                                           int embedding_num_pack,
                                           Param<T, IDX, pack_size, N> param) {
#pragma unroll 1
  for (int i = 0; i < parallel_num; ++i) {
    int rank_id = (parallel_id + i) % parallel_num;
    IDX out_index_offset = 0;
    for (int k = 0; k < rank_id; ++k) {
      out_index_offset += param.num_unique_matrix[parallel_id * parallel_num + k];
    }
    IDX in_index_offset = 0;
    for (int k = 0; k < parallel_id; ++k) {
      in_index_offset += param.num_unique_matrix[k * parallel_num + rank_id];
    }
    const IDX* inverse_indices_ptr = param.inverse_indices[rank_id] + in_index_offset;
    const Pack<T, pack_size>* unique_embeddings_ptr = param.unique_embeddings[rank_id];
    Pack<T, pack_size>* embedding_ptr = param.embedding_ptr + out_index_offset * embedding_num_pack;
    const int copy_cnt =
        param.num_unique_matrix[parallel_id * parallel_num + rank_id] * embedding_num_pack;
    CUDA_1D_KERNEL_LOOP_T(int, j, copy_cnt) {
      int out_row_id = j / embedding_num_pack;
      int in_row_id = inverse_indices_ptr[out_row_id];
      int col_id = j - out_row_id * embedding_num_pack;
      embedding_ptr[j] = unique_embeddings_ptr[in_row_id * embedding_num_pack + col_id];
    }
  }
}

template<typename T, typename IDX, int pack_size, int N>
__global__ void EmbeddingShuffleCopyKernel(int parallel_id, int parallel_num,
                                           int embedding_num_pack,
                                           Param<T, IDX, pack_size, N> param) {
#pragma unroll 1
  for (int i = 0; i < parallel_num; ++i) {
    int rank_id = (parallel_id + i) % parallel_num;
    IDX out_index_offset = 0;
    for (int k = 0; k < rank_id; ++k) {
      out_index_offset += param.num_unique_matrix[parallel_id * parallel_num + k];
    }
    IDX in_index_offset = 0;
    for (int k = 0; k < parallel_id; ++k) {
      in_index_offset += param.num_unique_matrix[k * parallel_num + rank_id];
    }
    const Pack<T, pack_size>* unique_embeddings_ptr =
        param.unique_embeddings[rank_id] + in_index_offset * embedding_num_pack;
    Pack<T, pack_size>* embedding_ptr = param.embedding_ptr + out_index_offset * embedding_num_pack;
    const int copy_cnt =
        param.num_unique_matrix[parallel_id * parallel_num + rank_id] * embedding_num_pack;
    CUDA_1D_KERNEL_LOOP_T(int, j, copy_cnt) { embedding_ptr[j] = unique_embeddings_ptr[j]; }
  }
}

template<typename T, typename IDX, int pack_size>
__global__ void GatherKernel(int parallel_id, int parallel_num, int embedding_num_pack,
                             const IDX* num_unique_matrix, const IDX* inverse_indices,
                             const Pack<T, pack_size>* unique_embeddings,
                             Pack<T, pack_size>* gather_out_unique_embeddings) {
  int cur_rank_num_ids = 0;
  for (int i = 0; i < parallel_num; ++i) {
    cur_rank_num_ids += num_unique_matrix[i * parallel_num + parallel_id];
  }
  int out_cnt = cur_rank_num_ids * embedding_num_pack;
  CUDA_1D_KERNEL_LOOP_T(int, i, out_cnt) {
    int out_row_id = i / embedding_num_pack;
    int in_row_id = inverse_indices[out_row_id];
    int col_id = i - out_row_id * embedding_num_pack;
    gather_out_unique_embeddings[i] = unique_embeddings[in_row_id * embedding_num_pack + col_id];
  }
}

template<typename T, typename IDX, int pack_size, int N>
__global__ void BarrierKernel(int32_t parallel_id, int32_t parallel_num,
                              Param<T, IDX, pack_size, N> param) {
  int count = param.is_kernel_start[parallel_id][parallel_id];
  if (threadIdx.x < parallel_num) {
    volatile int32_t* start_f = param.is_kernel_start[parallel_id];
    volatile int32_t* remote_start_f = param.is_kernel_start[threadIdx.x];
    start_f[threadIdx.x] = count + 1;
    while (remote_start_f[parallel_id] < count + 1) {}
  }
}

struct IpcMemHandleOffset {
  cudaIpcMemHandle_t handle;
  int64_t offset;
};

bool DisableFuseGatherCopy() {
  return ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_P2P_DISABLE_FUSE_GATHER_COPY", false);
}

void GetPtrs(user_op::KernelComputeContext* ctx, std::vector<void*>* unique_embeddings_ptr,
             std::vector<void*>* inverse_indices_ptr, std::vector<void*>* is_kernel_start_ptr) {
  const int64_t num_ids =
      ctx->TensorDesc4ArgNameAndIndex("inverse_unique_partition_indices", 0)->shape().elem_cnt();
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  inverse_indices_ptr->at(parallel_id) =
      const_cast<void*>(ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0)->dptr());
  if (DisableFuseGatherCopy()) {
    unique_embeddings_ptr->at(parallel_id) =
        ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)->mut_dptr();
  } else {
    unique_embeddings_ptr->at(parallel_id) =
        const_cast<void*>(ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0)->dptr());
  }

  std::string name = ctx->op_name();
  {
    std::vector<IpcMemHandleOffset> push_handle_offset;
    push_handle_offset.resize(3);
    OF_CUDA_CHECK(cudaIpcGetMemHandle(&push_handle_offset.at(0).handle,
                                      unique_embeddings_ptr->at(parallel_id)));
    OF_CUDA_CHECK(cudaIpcGetMemHandle(&push_handle_offset.at(1).handle,
                                      inverse_indices_ptr->at(parallel_id)));
    OF_CUDA_CHECK(cudaIpcGetMemHandle(&push_handle_offset.at(2).handle,
                                      is_kernel_start_ptr->at(parallel_id)));

    cudaError_t (*func)(void*, CUpointer_attribute, CUdeviceptr);
    OF_CUDA_CHECK(
        cudaGetDriverEntryPoint("cuPointerGetAttribute", (void**)(&func), cudaEnableDefault));
    void* unique_embeddings_base;
    OF_CUDA_CHECK(func(&unique_embeddings_base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                       (CUdeviceptr)(unique_embeddings_ptr->at(parallel_id))));
    push_handle_offset.at(0).offset =
        reinterpret_cast<char*>(unique_embeddings_ptr->at(parallel_id))
        - reinterpret_cast<char*>(unique_embeddings_base);
    void* inverse_indices_base;
    OF_CUDA_CHECK(func(&inverse_indices_base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                       (CUdeviceptr)(inverse_indices_ptr->at(parallel_id))));
    push_handle_offset.at(1).offset = reinterpret_cast<char*>(inverse_indices_ptr->at(parallel_id))
                                      - reinterpret_cast<char*>(inverse_indices_base);
    push_handle_offset.at(2).offset = 0;
    Singleton<CtrlClient>::Get()->PushKV(
        name + std::to_string(parallel_id),
        std::string(reinterpret_cast<const char*>(push_handle_offset.data()),
                    3 * sizeof(IpcMemHandleOffset)));
  }
  for (int64_t i = 0; i < parallel_num; ++i) {
    std::string key = name + std::to_string(i);
    if (parallel_id != i) {
      std::vector<IpcMemHandleOffset> handle_offset;
      handle_offset.resize(3);
      Singleton<CtrlClient>::Get()->PullKV(key, [i, &handle_offset](const std::string& val) {
        memcpy(handle_offset.data(), val.data(), 3 * sizeof(IpcMemHandleOffset));
      });
      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&unique_embeddings_ptr->at(i), handle_offset.at(0).handle,
                                         cudaIpcMemLazyEnablePeerAccess));
      unique_embeddings_ptr->at(i) =
          reinterpret_cast<char*>(unique_embeddings_ptr->at(i)) + handle_offset.at(0).offset;

      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&inverse_indices_ptr->at(i), handle_offset.at(1).handle,
                                         cudaIpcMemLazyEnablePeerAccess));
      inverse_indices_ptr->at(i) =
          reinterpret_cast<char*>(inverse_indices_ptr->at(i)) + handle_offset.at(1).offset;

      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&is_kernel_start_ptr->at(i), handle_offset.at(2).handle,
                                         cudaIpcMemLazyEnablePeerAccess));
      is_kernel_start_ptr->at(i) =
          reinterpret_cast<char*>(is_kernel_start_ptr->at(i)) + handle_offset.at(2).offset;
    }
  }
}

template<typename IDX>
class DataShuffleKernelState final : public user_op::OpKernelState {
 public:
  explicit DataShuffleKernelState(user_op::KernelInitContext* ctx)
      : device_index_(-1),
        parallel_desc_(ctx->parallel_desc()),
        parallel_id_(ctx->parallel_ctx().parallel_id()) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    int64_t parallel_num = parallel_desc_.parallel_num();
    unique_embeddings_ptr_.resize(parallel_num);
    inverse_indices_ptr_.resize(parallel_num);
    is_kernel_start_ptr_.resize(parallel_num);
    size_t is_kernel_start_size = GetCudaAlignedSize(parallel_num * sizeof(int32_t));
    OF_CUDA_CHECK(cudaMalloc(&is_kernel_start_ptr_.at(parallel_id_), is_kernel_start_size));
    OF_CUDA_CHECK(cudaMemset(is_kernel_start_ptr_.at(parallel_id_), 0, is_kernel_start_size));
  }

  ~DataShuffleKernelState() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(is_kernel_start_ptr_.at(parallel_id_)));
  }

  std::vector<void*>* UniqueEmbeddings() { return &unique_embeddings_ptr_; }

  std::vector<void*>* InverseIndices() { return &inverse_indices_ptr_; }

  std::vector<void*>* IsKernelStart() { return &is_kernel_start_ptr_; }

 private:
  int device_index_;
  ParallelDesc parallel_desc_;
  int64_t parallel_id_;
  std::vector<void*> unique_embeddings_ptr_;
  std::vector<void*> inverse_indices_ptr_;
  std::vector<void*> is_kernel_start_ptr_;
};

template<typename T, typename IDX, int pack_size>
void LaunchKernel(user_op::KernelComputeContext* ctx, DataShuffleKernelState<IDX>* kernel_state) {
  const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
  user_op::Tensor* embeddings = ctx->Tensor4ArgNameAndIndex("embeddings", 0);
  const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
  DataType data_type = embeddings->data_type();
  Param<T, IDX, pack_size, 8> param;
  CHECK_LE(parallel_num, 8);
  param.embedding_ptr = reinterpret_cast<Pack<T, pack_size>*>(embeddings->mut_dptr<T>());
  for (int i = 0; i < parallel_num; ++i) {
    param.inverse_indices[i] = reinterpret_cast<IDX*>(kernel_state->InverseIndices()->at(i));
    param.unique_embeddings[i] =
        reinterpret_cast<Pack<T, pack_size>*>(kernel_state->UniqueEmbeddings()->at(i));
    param.is_kernel_start[i] = reinterpret_cast<int32_t*>(kernel_state->IsKernelStart()->at(i));
  }
  param.num_unique_matrix = reinterpret_cast<const uint32_t*>(num_unique_matrix->dptr());
  int64_t embedding_num_pack = embedding_size / pack_size;
  cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
  BarrierKernel<<<1, parallel_num, 0, cuda_stream>>>(parallel_id, parallel_num, param);
  const int num_blocks =
      2 * ctx->stream()->As<ep::CudaStream>()->device_properties().multiProcessorCount;

  if (DisableFuseGatherCopy()) {
    CHECK_EQ(kernel_state->UniqueEmbeddings()->at(parallel_id),
             ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)->dptr())
        << parallel_id;
    GatherKernel<<<num_blocks, 1024, 0, cuda_stream>>>(
        parallel_id, parallel_num, embedding_num_pack, param.num_unique_matrix,
        param.inverse_indices[parallel_id],
        reinterpret_cast<const Pack<T, pack_size>*>(
            ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0)->dptr()),
        param.unique_embeddings[parallel_id]);
    EmbeddingShuffleCopyKernel<<<num_blocks, 1024, 0, cuda_stream>>>(parallel_id, parallel_num,
                                                                     embedding_num_pack, param);
  } else {
    CHECK_EQ(kernel_state->UniqueEmbeddings()->at(parallel_id),
             ctx->Tensor4ArgNameAndIndex("cur_rank_embeddings", 0)->dptr())
        << parallel_id;
    EmbeddingShuffleCudaKernel<<<num_blocks, 1024, 0, cuda_stream>>>(parallel_id, parallel_num,
                                                                     embedding_num_pack, param);
  }
  if (!ctx->Attr<bool>("is_train")) {
    BarrierKernel<<<1, parallel_num, 0, cuda_stream>>>(
        parallel_id, parallel_num,
        param);  // if in eval, should add last barrier.
  }
}

}  // namespace

template<typename T, typename IDX>
class EmbeddingShuffleP2PKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  EmbeddingShuffleP2PKernel() : current_iter_(0) {}
  ~EmbeddingShuffleP2PKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<DataShuffleKernelState<IDX>>(ctx);
  }

  bool IsReadyForCapture(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                         const user_op::OpKernelCache* cache) const override {
    if (current_iter_ == 0) {
      return false;
    } else {
      return true;
    }
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    CHECK(!embedding::UseDynamicMemoryAllocation());
    CHECK(ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION",
                              false));  // only support skip last gather.
    CHECK(ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ADD_ID_SHUFFLE_COPY_OUT",
                              true));  // when no identity, every time the cur_rank_inverse_indices
                                       // will change becauseof regster num=2.
    auto* kernel_state = dynamic_cast<DataShuffleKernelState<IDX>*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    const user_op::Tensor* inverse_unique_partition_indices =
        ctx->Tensor4ArgNameAndIndex("inverse_unique_partition_indices", 0);
    const bool skip_last_gather = ctx->Attr<bool>("skip_last_gather");
    CHECK(skip_last_gather);
    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    if (current_iter_ == 0) {
      GetPtrs(ctx, kernel_state->UniqueEmbeddings(), kernel_state->InverseIndices(),
              kernel_state->IsKernelStart());
    }
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    CHECK_EQ(kernel_state->InverseIndices()->at(parallel_id), cur_rank_inverse_indices->dptr())
        << parallel_id;
    if (embedding_size % 4 == 0) {
      LaunchKernel<T, IDX, 4>(ctx, kernel_state);
    } else if (embedding_size % 2 == 0) {
      LaunchKernel<T, IDX, 2>(ctx, kernel_state);
    } else {
      LaunchKernel<T, IDX, 1>(ctx, kernel_state);
    }
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

REGISTER_USER_KERNEL("embedding_shuffle")
    .SetCreateFn<EmbeddingShuffleP2PKernel<half, uint32_t>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("cur_rank_embeddings", 0) == DataType::kFloat16)
                     && (user_op::HobDataType("num_unique_matrix", 0) == DataType::kUInt32)
                     && (user_op::HobAttr<bool>("skip_last_gather") == true)
                     && (embedding::UseEmbeddingShuffleP2PKernel(DataType::kFloat16,
                                                                 DataType::kUInt32)))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      return GetCudaAlignedSize(ctx->InputTensorDesc("cur_rank_embeddings", 0).shape().elem_cnt()
                                * sizeof(half));
    });
}  // namespace oneflow

#endif  // CUDA_VERSION >= 11030
