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

namespace oneflow {

namespace {

template<typename T>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(int32_t capacity, void* ptr, const ShapeView& in_shape)
      : capacity_{capacity},
        sorted_in_elem_cnt_{in_shape.elem_cnt()},
        indices_elem_cnt_{sorted_in_elem_cnt_},
        sorted_indices_elem_cnt_{sorted_in_elem_cnt_} {
    const int32_t sorted_in_aligned_bytes = GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const int32_t indices_aligned_bytes = GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int32_t));
    const int32_t sorted_indices_aligned_bytes = indices_aligned_bytes;
    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    indices_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(sorted_in_ptr_)
                                              + sorted_in_aligned_bytes);
    sorted_indices_ptr_ =
        reinterpret_cast<int32_t*>(reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_ptr_ = reinterpret_cast<void*>(reinterpret_cast<char*>(sorted_indices_ptr_)
                                                + sorted_indices_aligned_bytes);
    temp_storage_bytes_ =
        capacity_ - sorted_in_aligned_bytes - indices_aligned_bytes - sorted_indices_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  T* SortedInPtr() const { return sorted_in_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  int32_t* SortedIndicesPtr() const { return sorted_indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t SortedInElemCnt() const { return sorted_in_elem_cnt_; }
  int32_t IndicesElemCnt() const { return indices_elem_cnt_; }
  int32_t SortedIndicesElemCnt() const { return sorted_indices_elem_cnt_; }
  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  T* sorted_in_ptr_;
  int32_t* indices_ptr_;
  int32_t* sorted_indices_ptr_;
  void* temp_storage_ptr_;

  int64_t sorted_in_elem_cnt_;
  int64_t indices_elem_cnt_;
  int64_t sorted_indices_elem_cnt_;
  int32_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int32_t elem_cnt, int32_t* indices_ptr, int32_t instance_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i % instance_size; };
}

}  // namespace

template<typename T>
class GpuRadixSortTopKKernel final : public user_op::OpKernel {
 public:
  GpuRadixSortTopKKernel() = default;
  ~GpuRadixSortTopKKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager<T> buf_manager(static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                    tmp_buffer->mut_dptr<void>(), in->shape());

    const int32_t elem_cnt = in->shape().elem_cnt();
    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = elem_cnt / instance_size;
    const int32_t k = std::min(ctx->Attr<int32_t>("k"), instance_size);
    InitializeIndices<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(elem_cnt, buf_manager.IndicesPtr(),
                                                            instance_size);
    SortPairsDescending(in->dptr<T>(), buf_manager.IndicesPtr(), instance_num, instance_size,
                        buf_manager.TempStoragePtr(), buf_manager.TempStorageBytes(),
                        buf_manager.SortedInPtr(), buf_manager.SortedIndicesPtr(),
                        ctx->device_ctx()->cuda_stream());
    OF_CUDA_CHECK(cudaMemcpy2DAsync(out->mut_dptr<int32_t>(), k * sizeof(int32_t),
                                    buf_manager.SortedIndicesPtr(), instance_size * sizeof(int32_t),
                                    k * sizeof(int32_t), instance_num, cudaMemcpyDefault,
                                    ctx->device_ctx()->cuda_stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_RADIX_SORT_TOP_K_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("top_k")                                                                    \
      .SetCreateFn<GpuRadixSortTopKKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") & (user_op::HobAttr<int32_t>("k") > 128) \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))             \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                          \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                               \
        const int32_t elem_cnt = in_shape->elem_cnt();                                             \
        const int32_t instance_size = in_shape->dim_vec().back();                                  \
        const int32_t instance_num = elem_cnt / instance_size;                                     \
                                                                                                   \
        /* Sorted In*/                                                                             \
        const int32_t sorted_in_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(dtype));      \
        /* Indices */                                                                              \
        const int32_t indices_aligned_bytes = GetCudaAlignedSize(elem_cnt * sizeof(int32_t));      \
        /* Sorted Indices */                                                                       \
        const int32_t sorted_indices_aligned_bytes = indices_aligned_bytes;                        \
        /* CUB Temp Storage */                                                                     \
        int32_t temp_storage_bytes =                                                               \
            InferTempStorageForSortPairsDescending<dtype, int32_t>(instance_num, instance_size);   \
                                                                                                   \
        return sorted_in_aligned_bytes + indices_aligned_bytes + sorted_indices_aligned_bytes      \
               + temp_storage_bytes;                                                               \
      });

REGISTER_GPU_RADIX_SORT_TOP_K_KERNEL(float)
REGISTER_GPU_RADIX_SORT_TOP_K_KERNEL(double)
REGISTER_GPU_RADIX_SORT_TOP_K_KERNEL(int32_t)
REGISTER_GPU_RADIX_SORT_TOP_K_KERNEL(int64_t)

}  // namespace oneflow
