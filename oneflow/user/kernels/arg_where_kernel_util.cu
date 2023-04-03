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
#include "oneflow/user/kernels/arg_where_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/small_vector.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

template<typename T, int NDIM>
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
    return *(ptr_ + (i * NDIM));
  }

 private:
  T* ptr_;
  size_t max_iters_;
};

template<typename T, int NDIM>
__global__ void __launch_bounds__(kBlockSize)
    CudaOffsetToNdIndexInplace(NdIndexOffsetHelper<T, NDIM> index_converter,
                               const T* output_size_ptr, T* output_ptr) {
  CUDA_1D_KERNEL_LOOP_T(T, i, *output_size_ptr) {
    T* index_ptr = output_ptr + i * NDIM;
    index_converter.OffsetToNdIndex(*index_ptr, index_ptr);
  }
}

template<typename T>
struct IsTrue {
  __device__ __forceinline__ bool operator()(const T& val) const { return static_cast<bool>(val); }
};

template<typename IN_T, typename OUT_T, typename OUT_ITER>
cudaError_t SelectTrue(cudaStream_t stream, int num_items, void* temp_storage,
                       size_t& temp_storage_bytes, const IN_T* input, OUT_ITER output_iter,
                       OUT_T* num_selected) {
  IsTrue<IN_T> is_true;
  cub::TransformInputIterator<bool, IsTrue<IN_T>, const IN_T*> flag_iter(input, is_true);
  cub::CountingInputIterator<OUT_T> offset_counter(0);
  return cub::DeviceSelect::Flagged(temp_storage, temp_storage_bytes, offset_counter, flag_iter,
                                    output_iter, num_selected, num_items, stream);
}

template<typename IN_T, typename OUT_T>
__global__ void SetOutputSizeKernel(const IN_T* input_ptr, OUT_T* output_size_ptr) {
  if (*input_ptr == GetZeroVal<IN_T>()) {
    *output_size_ptr = GetZeroVal<OUT_T>();
  } else {
    *output_size_ptr = GetOneVal<OUT_T>();
  }
}

}  // namespace

template<typename IN_T, typename OUT_T, int NDIM>
struct ArgWhereKernelUtil<DeviceType::kCUDA, IN_T, OUT_T, NDIM> {
  static void ArgWhere(ep::Stream* stream, const ShapeView& input_shape, const IN_T* input_ptr,
                       void* temp_storage, size_t temp_storage_bytes, OUT_T* output_ptr,
                       OUT_T* output_size_ptr) {
    const int64_t elem_cnt = input_shape.elem_cnt();
    // deal with empty blob
    if (elem_cnt == 0) {
      Memset<DeviceType::kCUDA>(stream, output_size_ptr, 0, sizeof(OUT_T));
      return;
    }

    CHECK_NOTNULL(stream);
    CHECK_LE(elem_cnt, std::numeric_limits<OUT_T>::max());
    size_t workspace = GetWorkspaceBytesSize(stream, elem_cnt);
    CHECK_LE(workspace, temp_storage_bytes);

    if (NDIM == 1) {
      OF_CUDA_CHECK((SelectTrue<IN_T, OUT_T, OUT_T*>(
          stream->As<ep::CudaStream>()->cuda_stream(), input_shape.elem_cnt(), temp_storage,
          workspace, input_ptr, output_ptr, output_size_ptr)));
    } else {
      using OutputIterator = StrideIterator<OUT_T, NDIM>;
      OutputIterator output_iter(output_ptr, elem_cnt);
      OF_CUDA_CHECK((SelectTrue<IN_T, OUT_T, OutputIterator>(
          stream->As<ep::CudaStream>()->cuda_stream(), elem_cnt, temp_storage, workspace, input_ptr,
          output_iter, output_size_ptr)));

      OUT_T dims[NDIM] = {0};
      std::transform(input_shape.ptr(), input_shape.ptr() + input_shape.NumAxes(), dims,
                     [](int64_t dim) { return static_cast<OUT_T>(dim); });
      NdIndexOffsetHelper<OUT_T, NDIM> index_converter(dims);
      CudaOffsetToNdIndexInplace<OUT_T, NDIM>
          <<<GetNumBlocks(elem_cnt), kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              index_converter, output_size_ptr, output_ptr);
    }
  }

  static size_t GetWorkspaceBytesSize(ep::Stream* stream, int64_t elem_cnt) {
    cudaStream_t cuda_stream = stream ? stream->As<ep::CudaStream>()->cuda_stream() : 0;
    size_t workspace = 0;
    if (NDIM == 1) {
      OF_CUDA_CHECK((SelectTrue<IN_T, OUT_T, OUT_T*>(cuda_stream, elem_cnt, nullptr, workspace,
                                                     nullptr, nullptr, nullptr)));
    } else {
      using OutputIterator = StrideIterator<OUT_T, NDIM>;
      OutputIterator output_iter(nullptr, elem_cnt);
      OF_CUDA_CHECK((SelectTrue<IN_T, OUT_T, OutputIterator>(
          cuda_stream, elem_cnt, nullptr, workspace, nullptr, output_iter, nullptr)));
    }
    return workspace;
  }
};

INSTANTIATE_ARG_WHERE_KERNEL_UTIL_FOR_DEVICE(DeviceType::kCUDA)

#define INSTANTIATE_CUDA_HALF_ARG_WHERE_KERNEL_UTIL                                              \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARG_WHERE_KERNEL_UTIL_WITH_DTYPE_PAIR,            \
                                   (DeviceType::kCUDA), HALF_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, \
                                   DIM_SEQ)

INSTANTIATE_CUDA_HALF_ARG_WHERE_KERNEL_UTIL

template<DeviceType device_type, typename IN_T, typename OUT_T>
void SetOutputSize(ep::Stream* stream, const IN_T* input_ptr, OUT_T* output_size_ptr) {
  SetOutputSizeKernel<IN_T, OUT_T>
      <<<1, 1, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(input_ptr, output_size_ptr);
}

INSTANTIATE_SET_OUTPUT_SIZE_FOR_DEVICE(DeviceType::kCUDA)

#define INSTANTIATE_CUDA_HALF_SET_OUTPUT_SIZE                                   \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SET_OUTPUT_SIZE_WITH_DTYPE_PAIR, \
                                   (DeviceType::kCUDA), HALF_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

INSTANTIATE_CUDA_HALF_SET_OUTPUT_SIZE

}  // namespace oneflow
