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
#include <cub/cub.cuh>
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(int32_t capacity, void* ptr, int32_t instance_num)
      : capacity_{capacity}, key_value_out_elem_cnt_{instance_num} {
    const int32_t key_value_out_aligned_bytes =
        GetCudaAlignedSize(key_value_out_elem_cnt_ * sizeof(cub::KeyValuePair<int32_t, T>));
    key_value_out_ptr_ = reinterpret_cast<cub::KeyValuePair<int32_t, T>*>(ptr);
    temp_storage_ptr_ = reinterpret_cast<void*>(reinterpret_cast<char*>(key_value_out_ptr_)
                                                + key_value_out_aligned_bytes);
    temp_storage_bytes_ = capacity_ - key_value_out_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  cub::KeyValuePair<int32_t, T>* KeyValueOutPtr() const { return key_value_out_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  cub::KeyValuePair<int32_t, T>* key_value_out_ptr_;
  void* temp_storage_ptr_;

  int32_t key_value_out_elem_cnt_;
  int32_t temp_storage_bytes_;
};

class MultiplyFunctor final {
 public:
  MultiplyFunctor(int32_t num_col) : num_col_(num_col) {}
  __host__ __device__ __forceinline__ int32_t operator()(int32_t idx) const {
    return idx * num_col_;
  }

 private:
  int32_t num_col_;
};

template<typename T>
size_t InferTempStorageForArgMax(int32_t num_row, int32_t num_col) {
  using SegmentOffsetIter =
      cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;
  cub::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  size_t temp_storage_bytes = 0;
  auto err =
      cub::DeviceSegmentedReduce::ArgMax<T*, cub::KeyValuePair<int32_t, T>*, SegmentOffsetIter>(
          /* d_temp_storage */ nullptr, /* temp_storage_bytes */ temp_storage_bytes,
          /* d_in */ nullptr, /* d_out */ nullptr, /* num_segments */ num_row,
          /* d_begin_offsets */ segment_offset_iter, /* d_end_offsets */ segment_offset_iter + 1,
          /* stream */ 0);
  OF_CUDA_CHECK(err);

  return temp_storage_bytes;
}

template<typename T>
void ArgMax(const T* in_ptr, int32_t num_row, int32_t num_col, void* temp_storage_ptr,
            int32_t temp_storage_bytes, cub::KeyValuePair<int32_t, T>* out_ptr,
            cudaStream_t stream) {
  size_t rt_inferred_temp_storage_bytes = InferTempStorageForArgMax<T>(num_row, num_col);
  CHECK_LE(rt_inferred_temp_storage_bytes, temp_storage_bytes);

  using SegmentOffsetIter =
      cub::TransformInputIterator<int32_t, MultiplyFunctor, cub::CountingInputIterator<int32_t>>;
  cub::CountingInputIterator<int32_t> counting_iter(0);
  MultiplyFunctor multiply_functor(num_col);
  SegmentOffsetIter segment_offset_iter(counting_iter, multiply_functor);

  auto err = cub::DeviceSegmentedReduce::ArgMax(
      /* d_temp_storage */ temp_storage_ptr,
      /* temp_storage_bytes */ rt_inferred_temp_storage_bytes,
      /* d_in */ in_ptr,
      /* d_out */ out_ptr,
      /* num_segments */ num_row,
      /* d_begin_offsets */ segment_offset_iter,
      /* d_end_offsets */ segment_offset_iter + 1,
      /* stream */ stream);
  OF_CUDA_CHECK(err);
}

template<typename T>
__global__ void WriteKeysToOutput(const int32_t instance_num,
                                  const cub::KeyValuePair<int32_t, T>* key_value_out_ptr,
                                  int64_t* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) { out_ptr[i] = key_value_out_ptr[i].key; }
}

}  // namespace

template<typename T>
class GpuArgMaxKernel final : public user_op::OpKernel {
 public:
  GpuArgMaxKernel() = default;
  ~GpuArgMaxKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int32_t elem_cnt = in->shape_view().elem_cnt();
    CHECK_GE(elem_cnt, 0);
    if (elem_cnt == 0) { return; }

    const int32_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int32_t instance_num = elem_cnt / instance_size;
    TmpBufferManager<T> buffer_manager(tmp_buffer->shape_view().elem_cnt(),
                                       tmp_buffer->mut_dptr<void>(), instance_num);

    ArgMax(in->dptr<T>(), instance_num, instance_size, buffer_manager.TempStoragePtr(),
           buffer_manager.TempStorageBytes(), buffer_manager.KeyValueOutPtr(),
           ctx->stream()->As<ep::CudaStream>()->cuda_stream());
    WriteKeysToOutput<T><<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0,
                           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        instance_num, buffer_manager.KeyValueOutPtr(), out->mut_dptr<int64_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ARGMAX_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("argmax")                                                                   \
      .SetCreateFn<GpuArgMaxKernel<dtype>>()                                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                             \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))            \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                          \
        const Shape& in_shape = ctx->InputShape("in", 0);                                          \
        const int32_t instance_size = in_shape.dim_vec().back();                                   \
        const int32_t instance_num = in_shape.elem_cnt() / instance_size;                          \
                                                                                                   \
        /* Key-Value Out */                                                                        \
        int32_t key_value_out_bytes =                                                              \
            GetCudaAlignedSize(instance_num * sizeof(cub::KeyValuePair<int32_t, dtype>));          \
                                                                                                   \
        /* CUB Temp Storage */                                                                     \
        size_t temp_storage_bytes = InferTempStorageForArgMax<dtype>(instance_num, instance_size); \
                                                                                                   \
        return key_value_out_bytes + temp_storage_bytes;                                           \
      });

REGISTER_CUDA_ARGMAX_KERNEL(bool)
REGISTER_CUDA_ARGMAX_KERNEL(float)
REGISTER_CUDA_ARGMAX_KERNEL(double)
REGISTER_CUDA_ARGMAX_KERNEL(uint8_t)
REGISTER_CUDA_ARGMAX_KERNEL(int8_t)
REGISTER_CUDA_ARGMAX_KERNEL(int32_t)
REGISTER_CUDA_ARGMAX_KERNEL(int64_t)
REGISTER_CUDA_ARGMAX_KERNEL(half)

}  // namespace oneflow
