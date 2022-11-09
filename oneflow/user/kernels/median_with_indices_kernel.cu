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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/radix_sort.cuh"

namespace oneflow {

namespace {

template<typename T, typename IDX>
__global__ void MedianSelectCuda(const IDX reduce_elem_cnt, const IDX stride, const T* in,
                                 const int64_t* sort_indices, T* values, int64_t* indices) {
  IDX nth = (stride - 1) / 2;
  CUDA_1D_KERNEL_LOOP_T(IDX, i, reduce_elem_cnt) {
    values[i] = in[i * stride + nth];
    indices[i] = sort_indices[i * stride + nth];
  }
}

bool IsSafeUseIndex32(int64_t elem_cnt) { return elem_cnt < GetMaxVal<int32_t>() / 2; }

template<typename T>
void DispatchIndexSize(ep::Stream* stream, const int64_t elem_cnt, const int64_t stride,
                       const T* in, const int64_t* sort_indices, T* out, int64_t* out_indices) {
  const int64_t reduce_elem_cnt = elem_cnt / stride;
  if (IsSafeUseIndex32(elem_cnt)) {
    RUN_CUDA_KERNEL((MedianSelectCuda<T, int32_t>), stream, reduce_elem_cnt, reduce_elem_cnt,
                    stride, in, sort_indices, out, out_indices);
  } else {
    RUN_CUDA_KERNEL((MedianSelectCuda<T, int64_t>), stream, reduce_elem_cnt, reduce_elem_cnt,
                    stride, in, sort_indices, out, out_indices);
  }
}

template<typename T>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(size_t capacity, void* ptr, const ShapeView& in_shape)
      : capacity_{capacity},
        sorted_in_elem_cnt_{in_shape.elem_cnt()},
        indices_elem_cnt_{sorted_in_elem_cnt_} {
    const size_t sort_tensor_buffer_bytes = GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const size_t sort_indices_buffer_bytes =
        GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int64_t));
    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    in_indices_ptr_ = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(sorted_in_ptr_)
                                                 + sort_tensor_buffer_bytes);
    out_indices_ptr_ = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(in_indices_ptr_)
                                                  + sort_indices_buffer_bytes);
    temp_storage_ptr_ = reinterpret_cast<void*>(reinterpret_cast<char*>(out_indices_ptr_)
                                                + sort_indices_buffer_bytes);
    temp_storage_bytes_ = capacity_ - sort_tensor_buffer_bytes - sort_indices_buffer_bytes * 2;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  T* SortedInPtr() const { return sorted_in_ptr_; }
  int64_t* InIndicesPtr() const { return in_indices_ptr_; }
  int64_t* OutIndicesPtr() const { return out_indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  size_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  size_t capacity_;

  T* sorted_in_ptr_;
  int64_t* in_indices_ptr_;
  int64_t* out_indices_ptr_;
  void* temp_storage_ptr_;

  int64_t sorted_in_elem_cnt_;
  int64_t indices_elem_cnt_;
  size_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int64_t elem_cnt, int64_t* indices_ptr, int64_t instance_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i % instance_size; };
}

}  // namespace

template<typename T>
class CudaMedianWithIndicesKernel final : public user_op::OpKernel {
 public:
  CudaMedianWithIndicesKernel() = default;
  ~CudaMedianWithIndicesKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    if (in->shape_view().elem_cnt() == 0) return;
    user_op::Tensor* values = ctx->Tensor4ArgNameAndIndex("values", 0);
    user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager<T> buf_manager(tmp_buffer->shape_view().elem_cnt(),
                                    tmp_buffer->mut_dptr<void>(), in->shape_view());

    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int64_t instance_num = elem_cnt / instance_size;
    RUN_CUDA_KERNEL(InitializeIndices, ctx->stream(), elem_cnt, elem_cnt,
                    buf_manager.InIndicesPtr(), instance_size);
    SortPairsAscending(in->dptr<T>(), buf_manager.InIndicesPtr(), instance_num, instance_size,
                       buf_manager.TempStoragePtr(), buf_manager.TempStorageBytes(),
                       buf_manager.SortedInPtr(), buf_manager.OutIndicesPtr(),
                       ctx->stream()->As<ep::CudaStream>()->cuda_stream());
    DispatchIndexSize(ctx->stream(), elem_cnt, instance_size, buf_manager.SortedInPtr(),
                      buf_manager.OutIndicesPtr(), values->mut_dptr<T>(),
                      indices->mut_dptr<int64_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_MEDIAN_WITH_INDICES_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("median_with_indices")                                                      \
      .SetCreateFn<CudaMedianWithIndicesKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                             \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value))         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                                \
        const Shape& in_shape = ctx->InputShape("input", 0);                                       \
        const int64_t instance_size = in_shape.dim_vec().back();                                   \
        const int64_t instance_num = in_shape.elem_cnt() / instance_size;                          \
        size_t sort_tmp_buffer_bytes =                                                             \
            InferTempStorageForSortPairsAscending<dtype, int64_t>(instance_num, instance_size);    \
        size_t sort_tensor_buffer_bytes = GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(dtype)); \
        size_t sort_indices_buffer_bytes =                                                         \
            GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(int64_t));                             \
        return sort_tmp_buffer_bytes + sort_tensor_buffer_bytes + sort_indices_buffer_bytes * 2;   \
      });

REGISTER_CUDA_MEDIAN_WITH_INDICES_KERNEL(float)
REGISTER_CUDA_MEDIAN_WITH_INDICES_KERNEL(double)
REGISTER_CUDA_MEDIAN_WITH_INDICES_KERNEL(int8_t)
REGISTER_CUDA_MEDIAN_WITH_INDICES_KERNEL(uint8_t)
REGISTER_CUDA_MEDIAN_WITH_INDICES_KERNEL(int32_t)
REGISTER_CUDA_MEDIAN_WITH_INDICES_KERNEL(int64_t)

}  // namespace oneflow
