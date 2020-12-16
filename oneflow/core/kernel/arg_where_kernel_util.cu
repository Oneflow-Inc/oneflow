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
#include "oneflow/core/kernel/arg_where_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/fixed_vector.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int kFlatIndexToNdIndexProposedLaunchBlocks = 128;

template<typename T, size_t NDims>
struct StrideIterator {
  typedef StrideIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
  typedef std::random_access_iterator_tag iterator_category;

  explicit StrideIterator(T* ptr, size_t max_iters) : ptr_(ptr), max_iters_(max_iters) {}

  OF_DEVICE_FUNC reference operator[](int i) {
    assert(0 <= i && i < max_iters_);
    return *(ptr_ + (i * NDims));
  }

 private:
  T* ptr_;
  size_t max_iters_;
};

template<typename T, size_t NDims>
__global__ void CudaOffsetToNdIndexInplace(NdIndexOffsetHelper<T, NDims> index_converter,
                                           const T* num_indices_ptr, T* indices_ptr) {
  CUDA_1D_KERNEL_LOOP_T(T, i, *num_indices_ptr) {
    T* cur_indices_ptr = indices_ptr + i * NDims;
    index_converter.OffsetToNdIndex(*cur_indices_ptr, cur_indices_ptr);
  }
}

template<typename T>
struct IsTrue {
  OF_DEVICE_FUNC bool operator()(const T& val) const { return static_cast<bool>(val); }
};

template<typename T, typename I, typename Iter>
cudaError_t SelectTrue(cudaStream_t stream, int num_items, void* tmp, size_t& tmp_bytes,
                       const T* flags, Iter out_iter, I* num_selected) {
  IsTrue<T> is_true;
  cub::TransformInputIterator<bool, IsTrue<T>, const T*> flag_iter(flags, is_true);
  cub::CountingInputIterator<I> offset_counter(0);
  return cub::DeviceSelect::Flagged(tmp, tmp_bytes, offset_counter, flag_iter, out_iter,
                                    num_selected, num_items, stream, false);
}

}  // namespace

template<typename T, typename I, size_t NDims>
struct ArgWhereKernelUtil<DeviceType::kGPU, T, I, NDims> {
  static void ArgWhere(DeviceCtx* ctx, const ShapeView& in_shape, const T* in_ptr, void* tmp,
                       size_t tmp_max_bytes, I* out_ptr, I* out_size_ptr) {
    if (in_shape.elem_cnt() == 0) {  // deal with empty blob
      KernelUtil<DeviceType::kGPU, I>::Set(ctx, static_cast<I>(0), out_size_ptr);
      return;
    }
    CHECK_NOTNULL(ctx);
    CHECK_LE(in_shape.elem_cnt(), std::numeric_limits<I>::max());
    size_t tmp_bytes = GetArgWhereWorkspaceSizeInBytes(ctx, in_shape.elem_cnt());
    CHECK_LE(tmp_bytes, tmp_max_bytes);

    if (NDims == 1) {
      OF_CUDA_CHECK((SelectTrue<T, I, I*>(ctx->cuda_stream(), in_shape.elem_cnt(), tmp, tmp_bytes,
                                          in_ptr, out_ptr, out_size_ptr)));
    } else {
      StrideIterator<I, NDims> out_iter(out_ptr, in_shape.elem_cnt());
      OF_CUDA_CHECK(
          (SelectTrue<T, I, StrideIterator<I, NDims>>(ctx->cuda_stream(), in_shape.elem_cnt(), tmp,
                                                      tmp_bytes, in_ptr, out_iter, out_size_ptr)));

      fixed_vector<I, NDims> dims(NDims);
      std::transform(in_shape.ptr(), in_shape.ptr() + in_shape.NumAxes(), dims.begin(),
                     [](int64_t dim) { return static_cast<I>(dim); });
      NdIndexOffsetHelper<I, NDims> index_converter(dims.data(), dims.size());
      CudaOffsetToNdIndexInplace<I, NDims>
          <<<kFlatIndexToNdIndexProposedLaunchBlocks, kCudaThreadsNumPerBlock, 0,
             ctx->cuda_stream()>>>(index_converter, out_size_ptr, out_ptr);
    }
  }

  static size_t GetArgWhereWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n) {
    cudaStream_t stream = ctx ? ctx->cuda_stream() : 0;
    size_t tmp_bytes = 0;
    if (NDims == 1) {
      OF_CUDA_CHECK(
          (SelectTrue<T, I, I*>(stream, n, nullptr, tmp_bytes, nullptr, nullptr, nullptr)));
    } else {
      StrideIterator<I, NDims> out_iter(nullptr, n);
      OF_CUDA_CHECK((SelectTrue<T, I, StrideIterator<I, NDims>>(stream, n, nullptr, tmp_bytes,
                                                                nullptr, out_iter, nullptr)));
    }
    return tmp_bytes;
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARG_WHERE_KERNEL_UTIL, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
