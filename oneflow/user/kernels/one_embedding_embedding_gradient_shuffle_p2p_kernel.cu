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

template<typename T, int32_t pack_size>
__device__ __inline__ void AtomicAdd(Pack<T, pack_size>* address, Pack<T, pack_size> val) {
#pragma unroll
  for (int i = 0; i < pack_size; ++i) {
    cuda::atomic::Add(reinterpret_cast<T*>(address) + i, static_cast<T>(val.elem[i]));
  }
}

template<>
__device__ __inline__ void AtomicAdd<half, 2>(Pack<half, 2>* address, Pack<half, 2> val) {
  half2 h2_val;
  h2_val.x = static_cast<half>(val.elem[0]);
  h2_val.y = static_cast<half>(val.elem[1]);
  cuda::atomic::Add(reinterpret_cast<half2*>(address), h2_val);
}

template<typename T, typename IDX, int pack_size, int N>
struct Param {
  const IDX* cur_rank_inverse_indices;
  const Pack<T, pack_size>* unique_partitioned_embedding_grads[N];
  int32_t* is_kernel_start[N];
  const IDX* num_unique_matrix;
  Pack<T, pack_size>* cur_rank_unique_embedding_grad_ptr;
};

template<typename T, typename IDX, int pack_size, int N>
__global__ void EmbeddingGradientShuffleCudaKernel(int64_t parallel_id, int64_t parallel_num,
                                                   int64_t embedding_num_pack,
                                                   Param<T, IDX, pack_size, N> param) {
#pragma unroll 1
  for (int i = 0; i < parallel_num; ++i) {
    int rank_id = (parallel_id + i) % parallel_num;
    IDX cur_rank_index_offset = 0;
    for (int k = 0; k < rank_id; ++k) {
      cur_rank_index_offset += param.num_unique_matrix[k * parallel_num + parallel_id];
    }
    IDX in_index_offset = 0;
    for (int k = 0; k < parallel_id; ++k) {
      in_index_offset += param.num_unique_matrix[rank_id * parallel_num + k];
    }
    const IDX* cur_rank_inverse_indices_ptr =
        param.cur_rank_inverse_indices + cur_rank_index_offset;
    const Pack<T, pack_size>* unique_partitioned_embedding_grad_ptr =
        param.unique_partitioned_embedding_grads[rank_id] + in_index_offset * embedding_num_pack;
    Pack<T, pack_size>* cur_rank_unique_embedding_grad_ptr =
        param.cur_rank_unique_embedding_grad_ptr;
    const int copy_cnt =
        param.num_unique_matrix[rank_id * parallel_num + parallel_id] * embedding_num_pack;
    CUDA_1D_KERNEL_LOOP_T(int, j, copy_cnt) {
      int in_row_id = j / embedding_num_pack;
      int col_id = j - in_row_id * embedding_num_pack;
      int out_row_id = cur_rank_inverse_indices_ptr[in_row_id];
      Pack<T, pack_size> grad_val = unique_partitioned_embedding_grad_ptr[j];
      AtomicAdd(cur_rank_unique_embedding_grad_ptr + out_row_id * embedding_num_pack + col_id,
                grad_val);
    }
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

void GetPtrs(user_op::KernelComputeContext* ctx,
             std::vector<void*>* unique_partitioned_embedding_grad_ptr,
             std::vector<void*>* is_kernel_start_ptr) {
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  unique_partitioned_embedding_grad_ptr->at(parallel_id) =
      const_cast<void*>(ctx->Tensor4ArgNameAndIndex("embedding_grad", 0)->dptr());
  std::string name = ctx->op_name();
  {
    std::vector<IpcMemHandleOffset> push_handle_offset;
    push_handle_offset.resize(2);
    OF_CUDA_CHECK(cudaIpcGetMemHandle(&push_handle_offset.at(0).handle,
                                      unique_partitioned_embedding_grad_ptr->at(parallel_id)));
    OF_CUDA_CHECK(cudaIpcGetMemHandle(&push_handle_offset.at(1).handle,
                                      is_kernel_start_ptr->at(parallel_id)));
    cudaError_t (*func)(void*, CUpointer_attribute, CUdeviceptr);
    OF_CUDA_CHECK(
        cudaGetDriverEntryPoint("cuPointerGetAttribute", (void**)(&func), cudaEnableDefault));
    void* embedding_grad_base;
    OF_CUDA_CHECK(func(&embedding_grad_base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                       (CUdeviceptr)(unique_partitioned_embedding_grad_ptr->at(parallel_id))));
    push_handle_offset.at(0).offset =
        reinterpret_cast<char*>(unique_partitioned_embedding_grad_ptr->at(parallel_id))
        - reinterpret_cast<char*>(embedding_grad_base);
    push_handle_offset.at(1).offset = 0;
    Singleton<CtrlClient>::Get()->PushKV(
        name + std::to_string(parallel_id),
        std::string(reinterpret_cast<const char*>(push_handle_offset.data()),
                    2 * sizeof(IpcMemHandleOffset)));
  }
  for (int64_t i = 0; i < parallel_num; ++i) {
    std::string key = name + std::to_string(i);
    if (parallel_id != i) {
      std::vector<IpcMemHandleOffset> handle_offset;
      handle_offset.resize(2);
      Singleton<CtrlClient>::Get()->PullKV(key, [i, &handle_offset](const std::string& val) {
        memcpy(handle_offset.data(), val.data(), 2 * sizeof(IpcMemHandleOffset));
      });
      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&unique_partitioned_embedding_grad_ptr->at(i),
                                         handle_offset.at(0).handle,
                                         cudaIpcMemLazyEnablePeerAccess));
      unique_partitioned_embedding_grad_ptr->at(i) =
          reinterpret_cast<char*>(unique_partitioned_embedding_grad_ptr->at(i))
          + handle_offset.at(0).offset;
      OF_CUDA_CHECK(cudaIpcOpenMemHandle(&is_kernel_start_ptr->at(i), handle_offset.at(1).handle,
                                         cudaIpcMemLazyEnablePeerAccess));
      is_kernel_start_ptr->at(i) =
          reinterpret_cast<char*>(is_kernel_start_ptr->at(i)) + handle_offset.at(1).offset;
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
    unique_partitioned_embedding_grad_ptr_.resize(parallel_num);
    is_kernel_start_ptr_.resize(parallel_num);
    size_t is_kernel_start_size = GetCudaAlignedSize(parallel_num * sizeof(int32_t));
    OF_CUDA_CHECK(cudaMalloc(&is_kernel_start_ptr_.at(parallel_id_), is_kernel_start_size));
    OF_CUDA_CHECK(cudaMemset(is_kernel_start_ptr_.at(parallel_id_), 0, is_kernel_start_size));
  }

  ~DataShuffleKernelState() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFree(is_kernel_start_ptr_.at(parallel_id_)));
  }

  std::vector<void*>* UniquePartitionedEmbeddingGrads() {
    return &unique_partitioned_embedding_grad_ptr_;
  }

  std::vector<void*>* IsKernelStart() { return &is_kernel_start_ptr_; }

 private:
  int device_index_;
  ParallelDesc parallel_desc_;
  int64_t parallel_id_;
  std::vector<void*> unique_partitioned_embedding_grad_ptr_;
  std::vector<void*> is_kernel_start_ptr_;
};

constexpr int pack_size = 2;

template<typename T, size_t pack>
__global__ void MemsetCurRankEmbeddingGrad(int64_t parallel_id, int64_t parallel_num,
                                           int64_t vector_size, const uint32_t* num_unique_matrix,
                                           T* dst) {
  size_t count = 0;
  for (int i = 0; i < parallel_num; ++i) {
    count += num_unique_matrix[i * parallel_num + parallel_id] * vector_size;
  }
  const size_t pack_count = count / pack;
  Pack<T, pack> pack_value;
  for (int i = 0; i < pack; ++i) { pack_value.elem[i] = static_cast<T>(0); }
  auto* pack_dst = reinterpret_cast<Pack<T, pack>*>(dst);
  CUDA_1D_KERNEL_LOOP_T(size_t, i, pack_count) { pack_dst[i] = pack_value; }
  T* tail_dst = dst + pack_count * pack;
  const size_t tail_count = count - pack_count * pack;
  CUDA_1D_KERNEL_LOOP_T(size_t, i, tail_count) { tail_dst[i] = static_cast<T>(0); }
}

template<typename T, size_t pack>
typename std::enable_if<(pack != 0), void>::type LaunchPackMemsetCurRankEmbeddingGrad(
    cudaStream_t stream, const uint32_t* num_unique_matrix, T* ptr, int sm_count,
    int64_t vector_size, int64_t parallel_id, int64_t parallel_num) {
  MemsetCurRankEmbeddingGrad<T, pack><<<2 * sm_count, 1024, 0, stream>>>(
      parallel_id, parallel_num, vector_size, num_unique_matrix, ptr);
}

template<typename T, size_t pack>
typename std::enable_if<(pack == 0), void>::type LaunchPackMemsetCurRankEmbeddingGrad(
    cudaStream_t stream, const uint32_t* num_unique_matrix, T* ptr, int sm_count,
    int64_t vector_size, int64_t parallel_id, int64_t parallel_num) {
  LOG(FATAL) << "wrong alignment";
}

template<typename T>
void LaunchMemsetCurRankEmbeddingGrad(cudaStream_t stream, int sm_count, int64_t vector_size,
                                      int64_t parallel_id, int64_t parallel_num,
                                      const uint32_t* num_unique_matrix, T* ptr) {
  auto uintptr = reinterpret_cast<std::uintptr_t>(ptr);
  if (uintptr % 16 == 0) {
    LaunchPackMemsetCurRankEmbeddingGrad<T, 16 / sizeof(T)>(
        stream, num_unique_matrix, ptr, sm_count, vector_size, parallel_id, parallel_num);
  } else if (uintptr % 8 == 0) {
    LaunchPackMemsetCurRankEmbeddingGrad<T, 8 / sizeof(T)>(stream, num_unique_matrix, ptr, sm_count,
                                                           vector_size, parallel_id, parallel_num);
  } else if (uintptr % 4 == 0) {
    LaunchPackMemsetCurRankEmbeddingGrad<T, 4 / sizeof(T)>(stream, num_unique_matrix, ptr, sm_count,
                                                           vector_size, parallel_id, parallel_num);
  } else if (uintptr % 2 == 0) {
    LaunchPackMemsetCurRankEmbeddingGrad<T, 2 / sizeof(T)>(stream, num_unique_matrix, ptr, sm_count,
                                                           vector_size, parallel_id, parallel_num);
  } else {
    LaunchPackMemsetCurRankEmbeddingGrad<T, 1 / sizeof(T)>(stream, num_unique_matrix, ptr, sm_count,
                                                           vector_size, parallel_id, parallel_num);
  }
}

}  // namespace

template<typename T, typename IDX>
class EmbeddingGraidientShuffleP2PKernel final : public user_op::OpKernel,
                                                 public user_op::CudaGraphSupport {
 public:
  EmbeddingGraidientShuffleP2PKernel() : current_iter_(0) {}
  ~EmbeddingGraidientShuffleP2PKernel() override = default;

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
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    const user_op::Tensor* num_unique_matrix = ctx->Tensor4ArgNameAndIndex("num_unique_matrix", 0);
    const user_op::Tensor* cur_rank_inverse_indices =
        ctx->Tensor4ArgNameAndIndex("cur_rank_inverse_indices", 0);
    user_op::Tensor* cur_rank_unique_embedding_grad =
        ctx->Tensor4ArgNameAndIndex("cur_rank_unique_embedding_grad", 0);

    const int64_t embedding_size = ctx->Attr<int64_t>("embedding_size");
    const bool only_zero_valid_grad = ctx->Attr<bool>("only_zero_valid_grad");
    const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const int sm_count =
        ctx->stream()->As<ep::CudaStream>()->device_properties().multiProcessorCount;
    const bool skip_first_scatter = ctx->Attr<bool>("skip_first_scatter");
    CHECK(skip_first_scatter);
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    if (current_iter_ == 0) {
      GetPtrs(ctx, kernel_state->UniquePartitionedEmbeddingGrads(), kernel_state->IsKernelStart());
    }
    CHECK_EQ(kernel_state->UniquePartitionedEmbeddingGrads()->at(parallel_id),
             embedding_grad->dptr());
    Param<T, IDX, pack_size, 8> param;
    CHECK_EQ(embedding_size % pack_size, 0);
    CHECK_LE(parallel_num, 8);
    param.cur_rank_unique_embedding_grad_ptr =
        reinterpret_cast<Pack<T, pack_size>*>(cur_rank_unique_embedding_grad->mut_dptr<T>());
    for (int i = 0; i < parallel_num; ++i) {
      param.unique_partitioned_embedding_grads[i] = reinterpret_cast<Pack<T, pack_size>*>(
          kernel_state->UniquePartitionedEmbeddingGrads()->at(i));
      param.is_kernel_start[i] = reinterpret_cast<int32_t*>(kernel_state->IsKernelStart()->at(i));
    }
    param.cur_rank_inverse_indices = reinterpret_cast<const IDX*>(cur_rank_inverse_indices->dptr());
    param.num_unique_matrix = reinterpret_cast<const uint32_t*>(num_unique_matrix->dptr());
    int64_t embedding_num_pack = embedding_size / pack_size;
    if (only_zero_valid_grad) {
      LaunchMemsetCurRankEmbeddingGrad(cuda_stream, sm_count, embedding_size, parallel_id,
                                       parallel_num,
                                       reinterpret_cast<const uint32_t*>(num_unique_matrix->dptr()),
                                       cur_rank_unique_embedding_grad->mut_dptr<T>());
    } else {
      OF_CUDA_CHECK(cudaMemsetAsync(
          cur_rank_unique_embedding_grad->mut_dptr(), 0,
          cur_rank_unique_embedding_grad->shape_view().elem_cnt() * sizeof(T), cuda_stream));
    }
    BarrierKernel<<<1, parallel_num, 0, cuda_stream>>>(parallel_id, parallel_num, param);
    const int num_blocks =
        2 * ctx->stream()->As<ep::CudaStream>()->device_properties().multiProcessorCount;
    EmbeddingGradientShuffleCudaKernel<<<num_blocks, 1024, 0, cuda_stream>>>(
        parallel_id, parallel_num, embedding_num_pack, param);
    current_iter_++;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  mutable int64_t current_iter_;
};

REGISTER_USER_KERNEL("embedding_gradient_shuffle")
    .SetCreateFn<EmbeddingGraidientShuffleP2PKernel<half, uint32_t>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("embedding_grad", 0) == DataType::kFloat16)
                     && (user_op::HobDataType("num_unique_matrix", 0) == DataType::kUInt32)
                     && (user_op::HobAttr<bool>("skip_first_scatter") == true)
                     && (embedding::UseEmbeddingGradientShuffleP2PKernel(DataType::kFloat16,
                                                                         DataType::kUInt32)));

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11030
