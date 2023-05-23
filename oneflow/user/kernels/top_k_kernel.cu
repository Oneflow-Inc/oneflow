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
#include "oneflow/user/kernels/radix_sort.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#include <cub/cub.cuh>
namespace oneflow {

namespace {

const float HALF_FLT_MAX = 65504.F;
const int kBlocksPerBatchStage1 = 8;

template<typename T>
struct NumericLimit {};

template<>
struct NumericLimit<half> { 
  __forceinline__ __device__ static half max(){ return static_cast<half>(HALF_FLT_MAX); } 
};
template<>
struct NumericLimit<float> { 
  __forceinline__ __device__ static float max(){ return static_cast<float>(FLT_MAX); } 
};

template<>
struct NumericLimit<double> { 
  __forceinline__ __device__ static double max(){ return static_cast<double>(DBL_MAX); } 
};

template<>
struct NumericLimit<uint8_t> { 
  __forceinline__ __device__ static uint8_t max(){ return static_cast<uint8_t>(USHRT_MAX); } 
};

template<>
struct NumericLimit<int8_t> { 
  __forceinline__ __device__ static int8_t max(){ return static_cast<int8_t>(SHRT_MAX); } 
};

template<>
struct NumericLimit<int32_t> { 
  __forceinline__ __device__ static int32_t max(){ return static_cast<int32_t>(INT_MAX); } 
};

template<>
struct NumericLimit<int64_t> { 
  __forceinline__ __device__ static int64_t max(){ return static_cast<int64_t>(LONG_MAX); } 
};

template<typename T, typename IndexType>
struct TopKReduceUnit {
    IndexType index = 0;
    T value = - NumericLimit<T>::max();

    __device__ __forceinline__ void insert(T elem, IndexType elem_id){
        if (elem > value) {
            value = elem;
            index = elem_id;
        }
    }

    __device__ __forceinline__ void init(){    
        value = - NumericLimit<T>::max();
        index = 0;
    }
};


template<typename T, typename IndexType>
__device__ __forceinline__ TopKReduceUnit<T, IndexType> reduce_topk_op(
  const TopKReduceUnit<T, IndexType>& a, const TopKReduceUnit<T, IndexType>& b){
    return a.value > b.value ? a : b;
}

template<typename T, typename IndexType, int kBlockSize, int kBlocksPerBatch>
__global__ void reduceTopKStage1(
    const T* input,             // m x n
    T* temp_input,              // m x n
    T* out_values,              // m x k
    T* temp_out_values,         // m x kBlocksPerBatch x k
    int64_t* out_indices,       // m x k
    int64_t* temp_out_indices,  // m x kBlocksPerBatch x k
    IndexType m,                // #batch
    IndexType n,                // batch size
    IndexType k                 // k
) {
  typedef cub::BlockReduce<TopKReduceUnit<T, IndexType>, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  IndexType elem_id, ite;
  const IndexType tid = threadIdx.x;
  const IndexType bid = blockIdx.x;
  const IndexType batch_id = bid / kBlocksPerBatch;
  const IndexType block_lane = bid % kBlocksPerBatch; 

  // where the current batch starts
  const IndexType batch_start_index = batch_id * n;

  // where the current block write out to temp_out_values and temp_out_indices
  const IndexType temp_out_start_index = batch_id * kBlocksPerBatch * k + block_lane * k;

  TopKReduceUnit<T, IndexType> partial;
  const T    MAX_T_VAL = NumericLimit<T>::max();

  // copy the origin values to the temp buffer
  for(elem_id=tid+block_lane*kBlockSize; elem_id<n; elem_id+=kBlockSize*kBlocksPerBatch){
      IndexType index = elem_id + batch_start_index;
      temp_input[index] = input[index];
  }

  // reduce for k times to find out top-ks within current data block
  for(ite=0; ite<k; ite++) {
    partial.init();

    // thread-wise reduce
#pragma unroll
    for(elem_id=tid+block_lane*kBlockSize; elem_id<n; elem_id+=kBlockSize*kBlocksPerBatch){
        IndexType index = elem_id + batch_start_index;
        partial.insert(temp_input[index], index);
    }

    // block-wise reduce
    TopKReduceUnit<T, IndexType> total = BlockReduce(temp_storage).Reduce(
      partial, reduce_topk_op<T, IndexType>
    );

    // write out the result, and change the input buffer
    if(tid == 0){
        const int index = temp_out_start_index + ite;
        temp_out_values[index] = total.value;
        temp_out_indices[index] = total.index;
        temp_input[total.index] = -MAX_T_VAL;
    }

    __syncthreads();
  }
}

template<typename T, typename IndexType, int kBlockSize, int kBlocksPerBatchStage1>
__global__ void reduceTopKStage2(
    const T* input,             // m x n
    T* temp_input,              // m x n
    T* out_values,              // m x k
    T* temp_out_values,         // m x kBlocksPerBatchStage1 x k
    int64_t* out_indices,       // m x k
    int64_t* temp_out_indices,  // m x kBlocksPerBatchStage1 x k
    IndexType m,                // #batch
    IndexType n,                // batch size
    IndexType k                 // k
) {
  typedef cub::BlockReduce<TopKReduceUnit<T, IndexType>, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage  temp_storage;

  const T    MAX_T_VAL = NumericLimit<T>::max();

  const IndexType tid      = threadIdx.x;
  const IndexType batch_id = blockIdx.x;

  IndexType elem_id, ite;

  // where the current block read from temp_out_values and temp_out_indices
  const IndexType batch_temp_start_index = batch_id * kBlocksPerBatchStage1 * k;

  // where the current block write to out_values and out_indices
  const IndexType batch_out_index = batch_id * k;

  TopKReduceUnit<T, IndexType> partial;

  for(ite=0; ite<k; ite++){
    partial.init();

    // thread-wise reduce
#pragma unroll
    for(elem_id=tid; elem_id<kBlocksPerBatchStage1*k; elem_id+=kBlockSize){
        const IndexType index = batch_temp_start_index + elem_id;
        partial.insert(temp_out_values[index], index);
    }

    // block-wise reduce
    TopKReduceUnit<T, IndexType> total = BlockReduce(temp_storage).Reduce(
      partial, reduce_topk_op<T, IndexType>
    );

    // write out the result, and change the input buffer
    if(tid == 0){
        const IndexType index = batch_out_index + ite;
        out_values[index] = total.value;
        out_indices[index] = temp_out_indices[total.index] % n;
        temp_out_values[total.index] = -MAX_T_VAL;
    }

    __syncthreads();
  }
} 

template<typename T, typename IndexType, int kBlockSizeStage1, int kBlockSizeStage2>
void LaunchKernel(
  ep::Stream* stream,
  const T* input,             // m x n
  T* temp_input,              // m x n
  T* out_values,              // m x k
  T* temp_out_values,         // m x kBlocksPerBatch x k
  int64_t* out_indices,       // m x k
  int64_t* temp_out_indices,  // m x kBlocksPerBatch x k
  IndexType m,                // #batch
  IndexType n,                // batch size
  IndexType k                 // k
){
  /* stage 1 reducing */
  reduceTopKStage1<T, IndexType, kBlockSizeStage1, kBlocksPerBatchStage1>
    <<<m * kBlocksPerBatchStage1, kBlockSizeStage1, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      /* input */ input,
      /* temp_input */ temp_input,
      /* out_values */ out_values,
      /* temp_out_values */ temp_out_values,
      /* out_indices */ out_indices,
      /* temp_out_indices */ temp_out_indices,
      /* m */ m,
      /* n */ n,
      /* k */ k
  );
  CHECK_JUST(stream->Sync());

  /* stage 2 reducing */
  reduceTopKStage2<T, IndexType, kBlockSizeStage2, kBlocksPerBatchStage1>
    <<<m, kBlockSizeStage2, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
      /* input */ input,
      /* temp_input */ temp_input,
      /* out_values */ out_values,
      /* temp_out_values */ temp_out_values,
      /* out_indices */ out_indices,
      /* temp_out_indices */ temp_out_indices,
      /* m */ m,
      /* n */ n,
      /* k */ k
  );
  CHECK_JUST(stream->Sync());
}

template<typename T, typename IndexType>
void DispatchBlockSize(
  ep::Stream* stream,
  const T* input,             // m x n
  T* temp_input,              // m x n
  T* out_values,              // m x k
  T* temp_out_values,         // m x kBlocksPerBatch x k
  int64_t* out_indices,       // m x k
  int64_t* temp_out_indices,  // m x kBlocksPerBatch x k
  IndexType m,                // #batch
  IndexType n,                // batch size
  IndexType k                 // k
){

#define LAUNCH_KERNEL(block_size_s1, block_size_s2)           \
  LaunchKernel<T, IndexType, block_size_s1, block_size_s2>(   \
    /* stream */ stream,                                      \
    /* input */ input,                                        \
    /* temp_input */ temp_input,                              \
    /* out_values */ out_values,                              \
    /* temp_out_values */ temp_out_values,                    \
    /* out_indices */ out_indices,                            \
    /* temp_out_indices */ temp_out_indices,                  \
    /* m */ m,                                                \
    /* n */ n,                                                \
    /* k */ k                                                 \
  );

  if (k >= 1 && k <= 16) {
    LAUNCH_KERNEL( /* block_size_s1 */128, /* block_size_s2 */128 );
  } else if (k >= 17 && k <= 32) {
    LAUNCH_KERNEL( /* block_size_s1 */256, /* block_size_s2 */128 );
  } else if (k >= 33 && k <= 64) {
    LAUNCH_KERNEL( /* block_size_s1 */256, /* block_size_s2 */256 );
  } else if (k >= 65 && k <= 1024) {
    LAUNCH_KERNEL( /* block_size_s1 */256, /* block_size_s2 */256 );
  } else {
    THROW(RuntimeError) << "top-k kernel can't support k that exceed 1024";
  }

#undef LAUNCH_KERNEL

}

template<typename T>
void DispatchIndexType(
  ep::Stream* stream,
  const T* input,             // m x n
  T* temp_input,              // m x n
  T* out_values,              // m x k
  T* temp_out_values,         // m x kBlocksPerBatch x k
  int64_t* out_indices,       // m x k
  int64_t* temp_out_indices,  // m x kBlocksPerBatch x k
  int64_t m,                  // #batch
  int64_t n,                  // batch size
  int64_t k                   // k
){
  const int64_t nb_input_elements = m * n;
  const int64_t nb_output_elements = m * k;

#define DISPATCH_BLOCK_SIZE(index_type)           \
    DispatchBlockSize<T, index_type>(             \
      /* stream */ stream,                        \
      /* input */ input,                          \
      /* temp_input */ temp_input,                \
      /* out_values */ out_values,                \
      /* temp_out_values */ temp_out_values,      \
      /* out_indices */ out_indices,              \
      /* temp_out_indices */ temp_out_indices,    \
      /* m */ static_cast<index_type>(m),         \
      /* n */ static_cast<index_type>(n),         \
      /* k */ static_cast<index_type>(k)          \
    );

  if (nb_input_elements < (1 << 30) && nb_output_elements < (1 << 30)) {
    DISPATCH_BLOCK_SIZE(int32_t);
  } else {
    DISPATCH_BLOCK_SIZE(int64_t);
  }

  #undef DISPATCH_BLOCK_SIZE
}

template<typename T, int kBlocksPerBatchStage1>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(int64_t capacity, void* ptr, const ShapeView& in_shape, const ShapeView& out_shape)
      : capacity_{capacity}, 
        temp_input_elem_cnt_{in_shape.elem_cnt()},
        temp_output_values_elem_cnt_{out_shape.elem_cnt() * kBlocksPerBatchStage1},
        temp_output_indices_elem_cnt_{temp_output_values_elem_cnt_} {
    const int64_t temp_input_aligned_bytes 
      = GetCudaAlignedSize(temp_input_elem_cnt_ * sizeof(T));
    const int64_t temp_output_values_aligned_bytes 
      = GetCudaAlignedSize(temp_output_values_elem_cnt_ * sizeof(T));
    const int64_t temp_output_indices_aligned_bytes 
      = GetCudaAlignedSize(temp_output_indices_elem_cnt_ * sizeof(int64_t));

    temp_input_ptr = reinterpret_cast<T*>(ptr);

    temp_output_values_ptr 
      = reinterpret_cast<T*>(reinterpret_cast<char*>(temp_input_ptr) + temp_input_aligned_bytes);

    temp_output_indices_ptr_ 
      = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(temp_output_values_ptr) 
        + temp_output_values_aligned_bytes);

    CHECK_GE(capacity,
      temp_input_aligned_bytes + temp_output_values_aligned_bytes + temp_output_indices_aligned_bytes
    );
  }

  ~TmpBufferManager() = default;

  T* TempInputPtr() const { return temp_input_ptr; }
  T* TempOutputValuesPtr() const { return temp_output_values_ptr; }
  int64_t* TempOutputIndicesPtr() const { return temp_output_indices_ptr_; }

 private:
  int64_t capacity_;
  int64_t temp_input_elem_cnt_;
  int64_t temp_output_values_elem_cnt_;
  int64_t temp_output_indices_elem_cnt_;
  T *temp_input_ptr;
  T *temp_output_values_ptr;
  int64_t *temp_output_indices_ptr_;
};

}  // namespace

template<typename T>
class GpuTopKKernel final : public user_op::OpKernel {
 public:
  GpuTopKKernel() = default;
  ~GpuTopKKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    user_op::Tensor* out_values = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* out_indices = ctx->Tensor4ArgNameAndIndex("indices", 0);

    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int64_t instance_num = elem_cnt / instance_size;
    const int64_t k = std::min(static_cast<int64_t>(ctx->Attr<int32_t>("k")), instance_size);
    
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager<T, kBlocksPerBatchStage1> buf_manager(
      /* capacity */ static_cast<int64_t>(tmp_buffer->shape_view().elem_cnt()),
      /* ptr */ tmp_buffer->mut_dptr<void>(),
      /* in_shape */ in->shape_view(),
      /* out_shape */ out_values->shape_view()
    );

    DispatchIndexType<T>(
      /* stream */ ctx->stream(),
      /* input */ in->dptr<T>(),
      /* temp_input */ buf_manager.TempInputPtr(),
      /* out_values */ out_values->mut_dptr<T>(),
      /* temp_out_values */ buf_manager.TempOutputValuesPtr(),
      /* out_indices */ out_indices->mut_dptr<int64_t>(),
      /* temp_out_indices */ buf_manager.TempOutputIndicesPtr(),
      /* m */ instance_num,
      /* n */ instance_size,
      /* k */ k
    );
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_TOP_K_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("top_k")                                                                   \
      .SetCreateFn<GpuTopKKernel<dtype>>()                                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                            \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))           \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const Shape& in_shape = ctx->InputShape("in", 0);                                         \
        const int64_t elem_cnt = in_shape.elem_cnt();                                             \
        const int64_t instance_size = in_shape.dim_vec().back();                                  \
        const int64_t k = std::min(static_cast<int64_t>(ctx->Attr<int32_t>("k")), instance_size); \
        const int64_t instance_num = elem_cnt / instance_size;                                    \
                                                                                                  \
        /* Temp Input */                                                                          \
        const int64_t temp_input_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(dtype));    \
                                                                                                  \
        /* Temp Output Values */                                                                  \
        const int64_t temp_output_values_aligned_bytes                                            \
            = GetCudaAlignedSize(instance_num * kBlocksPerBatchStage1 * k * sizeof(dtype));       \
                                                                                                  \
        /* Temp Output Indices */                                                                 \
        const int64_t temp_output_indices_aligned_bytes                                           \
            = GetCudaAlignedSize(instance_num * kBlocksPerBatchStage1 * k * sizeof(int64_t));     \
                                                                                                  \
        return temp_input_aligned_bytes + temp_output_values_aligned_bytes                        \
               + temp_output_indices_aligned_bytes;                                               \
      });

REGISTER_CUDA_TOP_K_KERNEL(float)
REGISTER_CUDA_TOP_K_KERNEL(double)
REGISTER_CUDA_TOP_K_KERNEL(uint8_t)
REGISTER_CUDA_TOP_K_KERNEL(int8_t)
REGISTER_CUDA_TOP_K_KERNEL(int32_t)
REGISTER_CUDA_TOP_K_KERNEL(int64_t)
REGISTER_CUDA_TOP_K_KERNEL(half)

}  // namespace oneflow
