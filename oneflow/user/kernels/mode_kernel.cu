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
#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/radix_sort.cuh"

namespace oneflow {

namespace {

template<typename Key>
__device__ __host__ __forceinline__ void print(Key* data, int64_t size) {
  FOR_RANGE(int64_t, i, 0, size) { printf("%f ", *(data + i)); }
}

template<typename InputType, typename OutputType>
size_t InferTempStorageForNonTrivialRuns(int32_t num_row, int32_t num_col) {
  size_t temp_storage_bytes = 0;
  auto err =
      cub::DeviceRunLengthEncode::NonTrivialRuns<InputType*, OutputType*, OutputType*, OutputType*>(
          /* d_temp_storage */ nullptr,
          /* temp_storage_bytes */ temp_storage_bytes,
          /* d_in */ nullptr,
          /* d_offsets_out  */ nullptr,
          /* d_lengths_out  */ nullptr,
          /* d_num_runs_out  */ nullptr,
          /* num_items */ num_row * num_col,
          /* stream */ 0);
  OF_CUDA_CHECK(err);
  return temp_storage_bytes;
}
template<typename Key, typename Index>
void NonTrivialsRun(Key* sorted_in, Index* sorted_indices, Key* out_values, Index* out_indices,
                    Index* offsets, Index* lengths, Index* nums, const int64_t& instance_num,
                    const int64_t& instance_size, cub::KeyValuePair<int32_t, Key>* d_argmax,
                    void* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
  FOR_RANGE(size_t, i, 0, instance_num) {
    Key* data = sorted_in + i * instance_size;
    Index* offset_out = offsets + i * instance_size;
    Index* length_out = lengths + i * instance_size;
    Index* num_out = nums + i;
    Key* out_value = out_values + i;
    Index* out_index = out_indices + i;
    Index* sorted_index = sorted_indices + i * instance_size;
    printf("%f ", *data);
    print<Key>(data, instance_size);
    cudaError_t err_non_trivial_run =
        cub::DeviceRunLengthEncode::NonTrivialRuns<Key*, Index*, Index*, Index*>(
            temp_storage, temp_storage_bytes, data, offset_out, length_out, num_out, instance_size,
            stream);
    OF_CUDA_CHECK(err_non_trivial_run);

    cudaError_t err_arg_max = cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes,
                                                        length_out, d_argmax, *num_out, stream);
    OF_CUDA_CHECK(err_arg_max);

    //TODO： 如果所有的数都只出现了一次，需要选择第一个数输出 这里需要加一个特例判断
    *out_value = data[(int)offset_out[(int)d_argmax->value]];
    *out_index = sorted_indices[(int)offset_out[(int)d_argmax->value]];  // TODO:调整输出的索引位置
  }
}

template<typename T>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(size_t capacity, void* ptr, const ShapeView& in_shape, size_t instance_num,
                   size_t instance_size)
      : capacity_{capacity},
        sorted_in_elem_cnt_{in_shape.elem_cnt()},
        indices_elem_cnt_{sorted_in_elem_cnt_} {
    const size_t sort_tensor_buffer_bytes = GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const size_t sort_indices_buffer_bytes =
        GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int64_t));
    sort_temp_storage_bytes_ =
        InferTempStorageForSortPairsAscending<T, int64_t>(instance_num, instance_size);

    const size_t non_trivials_run_offsets_buffer_bytes = sort_indices_buffer_bytes;
    const size_t non_trivials_run_lengths_buffer_bytes = sort_indices_buffer_bytes;
    const size_t non_trivials_run_nums_buffer_bytes =
        GetCudaAlignedSize(instance_num * sizeof(int64_t));
    non_trivials_run_temp_storage_bytes_ =
        ::oneflow::InferTempStorageForNonTrivialRuns<T, int64_t>(instance_num, instance_size);

    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    in_indices_ptr_ = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(sorted_in_ptr_)
                                                 + sort_tensor_buffer_bytes);
    out_indices_ptr_ = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(in_indices_ptr_)
                                                  + sort_indices_buffer_bytes);
    sort_temp_storage_ptr_ = reinterpret_cast<void*>(reinterpret_cast<char*>(out_indices_ptr_)
                                                     + sort_indices_buffer_bytes);

    non_trivials_run_offsets_ptr_ = reinterpret_cast<int64_t*>(
        reinterpret_cast<char*>(sort_temp_storage_ptr_) + sort_temp_storage_bytes_);
    non_trivials_run_lengths_ptr_ =
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(non_trivials_run_offsets_ptr_)
                                   + non_trivials_run_offsets_buffer_bytes);
    non_trivials_run_nums_ptr_ =
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(non_trivials_run_lengths_ptr_)
                                   + non_trivials_run_lengths_buffer_bytes);
    // non_trivials_run_temp_storage_ptr_ =
    // reinterpret_cast<void*>(reinterpret_cast<char*>(non_trivials_run_nums_ptr_)
    //                                             + non_trivials_run_nums_buffer_bytes);
    key_value_out_ptr_ = reinterpret_cast<cub::KeyValuePair<int32_t, T>*>(
        reinterpret_cast<char*>(non_trivials_run_lengths_ptr_)
        + non_trivials_run_nums_buffer_bytes);
  }
  ~TmpBufferManager() = default;

  T* SortedInPtr() const { return sorted_in_ptr_; }
  int64_t* InIndicesPtr() const { return in_indices_ptr_; }
  int64_t* OutIndicesPtr() const { return out_indices_ptr_; }
  void* SortTempStoragePtr() const { return sort_temp_storage_ptr_; }
  size_t SortTempStorageBytes() const { return sort_temp_storage_bytes_; }

  int64_t* NonTrivialsRunOffsetsPtr() const { return non_trivials_run_offsets_ptr_; }
  int64_t* NonTrivialsRunLengthsPtr() const { return non_trivials_run_lengths_ptr_; }
  int64_t* NonTrivialsRunNumsPtr() const { return non_trivials_run_nums_ptr_; }
  void* NonTrivialsRunTempStoragePtr() const {
    return non_trivials_run_temp_storage_ptr_;
  }  // TODO:这里可以合并多个辅助内存

  cub::KeyValuePair<int32_t, T>* KeyValueOutPtr() const { return key_value_out_ptr_; }

 private:
  size_t capacity_;

  T* sorted_in_ptr_;
  int64_t* in_indices_ptr_;
  int64_t* out_indices_ptr_;
  void* sort_temp_storage_ptr_;

  int64_t* non_trivials_run_offsets_ptr_;
  int64_t* non_trivials_run_lengths_ptr_;
  int64_t* non_trivials_run_nums_ptr_;
  void* non_trivials_run_temp_storage_ptr_;

  int64_t sorted_in_elem_cnt_;
  int64_t indices_elem_cnt_;
  size_t sort_temp_storage_bytes_;
  size_t non_trivials_run_temp_storage_bytes_;

  cub::KeyValuePair<int32_t, T>* key_value_out_ptr_;
};

__global__ void InitializeIndices(int64_t elem_cnt, int64_t* indices_ptr, int64_t instance_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i % instance_size; };
}
}  // namespace

template<typename T>
class CudaModeKernel final : public user_op::OpKernel {
 public:
  CudaModeKernel() = default;
  ~CudaModeKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    //  大概思路
    //  1. 排序
    //  2. 使用 cub::DeviceRunLengthEncode::NonTrivialRuns 获得每一个元素的数量
    //  https://nvlabs.github.io/cub/structcub_1_1_device_run_length_encode.html#aa2318dc7a69f28a8c47d417aaf53db3a
    //  3. 获取最大数量的元素的值和索引,并根据排序时的索引得到原始位置的索引

    // 难点:
    // 1. InferTmpSize 是否每个开辟了辅助内存的函数都需要将其内存size加上这里
    // 2. 核函数的启动和参数的设置（多维数组需要设置迭代器？？ 不太清楚
    // 还需要看一下其他算子的核函数的代码）
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    if (in->shape_view().elem_cnt() == 0) return;
    user_op::Tensor* values = ctx->Tensor4ArgNameAndIndex("values", 0);
    user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int64_t instance_num = elem_cnt / instance_size;
    TmpBufferManager<T> buf_manager(tmp_buffer->shape_view().elem_cnt(),
                                    tmp_buffer->mut_dptr<void>(), in->shape_view(), instance_num,
                                    instance_size);
    RUN_CUDA_KERNEL(InitializeIndices, ctx->stream(), elem_cnt, elem_cnt,
                    buf_manager.InIndicesPtr(), instance_size);
    SortPairsAscending(in->dptr<T>(), buf_manager.InIndicesPtr(), instance_num, instance_size,
                       buf_manager.SortTempStoragePtr(), buf_manager.SortTempStorageBytes(),
                       buf_manager.SortedInPtr(), buf_manager.OutIndicesPtr(),
                       ctx->stream()->As<ep::CudaStream>()->cuda_stream());

    NonTrivialsRun<T, int64_t>(
        buf_manager.SortedInPtr(), buf_manager.OutIndicesPtr(), values->mut_dptr<T>(),
        indices->mut_dptr<int64_t>(), buf_manager.NonTrivialsRunOffsetsPtr(),
        buf_manager.NonTrivialsRunLengthsPtr(), buf_manager.NonTrivialsRunNumsPtr(), instance_num,
        instance_size, buf_manager.KeyValueOutPtr(), buf_manager.SortTempStoragePtr(),
        buf_manager.SortTempStorageBytes(), ctx->stream()->As<ep::CudaStream>()->cuda_stream());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

// template<typename InputType, typename OutputType>
// size_t InferTempStorageForNonTrivialRuns(int32_t num_row, int32_t num_col) {
//   size_t temp_storage_bytes = 0;
//   auto err =
//       cub::DeviceRunLengthEncode::NonTrivialRuns<InputType*, OutputType*, OutputType*,
//       OutputType*>(
//           /* d_temp_storage */ nullptr,
//           /* temp_storage_bytes */ temp_storage_bytes,
//           /* d_in */ nullptr,
//           /* d_offsets_out  */ nullptr,
//           /* d_lengths_out  */ nullptr,
//           /* d_num_runs_out  */ nullptr,
//           /* num_items */ num_row * num_col,
//           /* stream */ 0);
//   OF_CUDA_CHECK(err);
//   return temp_storage_bytes;
// }

template<typename dtype>
size_t GetInferTmpSize(const Shape& in_shape, const int64_t& instance_num,
                       const int64_t& instance_size) {
  size_t sort_tmp_buffer_bytes =
      InferTempStorageForSortPairsAscending<dtype, int64_t>(instance_num, instance_size);
  size_t sort_tensor_buffer_bytes = GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(dtype));
  size_t sort_indices_buffer_bytes = GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(int64_t));

  size_t non_trivials_run_lengths_buffer_bytes =
      GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(int64_t));
  size_t non_trivials_run_offsets_buffer_bytes =
      GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(int64_t));
  size_t non_trivials_run_nums_buffer_bytes = GetCudaAlignedSize(instance_num * sizeof(int64_t));
  size_t non_trivials_run_buffer_bytes =
      InferTempStorageForNonTrivialRuns<dtype, int64_t>(instance_num, instance_size);

  return sort_tmp_buffer_bytes + sort_tensor_buffer_bytes + sort_indices_buffer_bytes * 2
         + non_trivials_run_lengths_buffer_bytes + non_trivials_run_offsets_buffer_bytes
         + non_trivials_run_nums_buffer_bytes + non_trivials_run_buffer_bytes;
}

#define REGISTER_CUDA_MODE_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("mode")                                                             \
      .SetCreateFn<CudaModeKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        const Shape& in_shape = ctx->InputShape("input", 0);                               \
        const int64_t instance_size = in_shape.dim_vec().back();                           \
        const int64_t instance_num = in_shape.elem_cnt() / instance_size;                  \
        return GetInferTmpSize<dtype>(in_shape, instance_num, instance_size);              \
      });

REGISTER_CUDA_MODE_KERNEL(float)
REGISTER_CUDA_MODE_KERNEL(double)
REGISTER_CUDA_MODE_KERNEL(int8_t)
REGISTER_CUDA_MODE_KERNEL(uint8_t)
REGISTER_CUDA_MODE_KERNEL(int32_t)
REGISTER_CUDA_MODE_KERNEL(int64_t)

}  // namespace oneflow
