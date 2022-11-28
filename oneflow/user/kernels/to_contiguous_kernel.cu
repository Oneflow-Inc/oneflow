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
#include <type_traits>
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace {

constexpr int32_t kThreadWorkSize = 4;
constexpr int32_t kNumThreads = 32 * 4;
constexpr int32_t get_min_threads_num() { return kNumThreads; }
constexpr int32_t get_block_work_size() { return kThreadWorkSize * kNumThreads; }
constexpr int32_t get_num_blocks(int64_t elem_cnt) {
  return (elem_cnt + get_block_work_size() - 1) / get_block_work_size();
}

struct StrideParam {
  int32_t stride[SHAPE_MAX_AXIS_SIZE];

  StrideParam(const int64_t* stride_vec, const size_t ndim) {
    for (size_t i = 0; i < ndim; ++i) { stride[i] = stride_vec[i]; }
  }
};

template<typename IndexType, size_t ndim>
__device__ __forceinline__ IndexType compute_index(IndexType out_offset,
                                                   const StrideParam& out_params,
                                                   const StrideParam& in_params) {
  IndexType in_offset = 0;
  IndexType remaining = out_offset;

#pragma unroll
  for (size_t i = 0; i < ndim; ++i) {
    const IndexType idx = static_cast<IndexType>(remaining / out_params.stride[i]);
    remaining -= idx * out_params.stride[i];
    in_offset += idx * in_params.stride[i];
  }
  return in_offset;
}

template<typename T, typename IndexType, size_t ndim>
__global__ void ToContiguousForwardGpuParallel(IndexType count, const StrideParam in_stride,
                                               const StrideParam out_stride, const T* in_dptr,
                                               T* out_dptr, const int32_t num_block_threads,
                                               const int32_t thread_work_size,
                                               const int32_t block_work_size) {
  IndexType remaining = count - block_work_size * blockIdx.x;
  IndexType idx = blockIdx.x;
  IndexType thread_idx = threadIdx.x;
#pragma unroll
  for (int32_t i = 0; i < thread_work_size; i++) {
    if (thread_idx >= remaining) { return; }
    IndexType out_idx = thread_idx + block_work_size * idx;
    IndexType in_idx = compute_index<IndexType, ndim>(out_idx, out_stride, in_stride);
    out_dptr[out_idx] = in_dptr[in_idx];
    thread_idx += num_block_threads;
  }
}

template<typename T, typename IndexType>
void LaunchToContiguousKernel(ep::Stream* stream, IndexType count, const size_t ndim,
                              IndexType block_size, const std::vector<int64_t>& in_stride,
                              const DimVector& out_stride, const char* in_dptr, char* out_dptr) {
  const int32_t num_blocks = get_num_blocks(count);
  constexpr int32_t num_threads = get_min_threads_num();
  constexpr int32_t block_work_size = get_block_work_size();
  StrideParam param_in_stride(in_stride.data(), ndim), param_out_stride(out_stride.data(), ndim);

  switch (ndim) {
#define TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(dim)                                             \
  case dim:                                                                                 \
    ToContiguousForwardGpuParallel<T, IndexType, dim>                                       \
        <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(      \
            count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),  \
            reinterpret_cast<T*>(out_dptr), num_threads, kThreadWorkSize, block_work_size); \
    break;

    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(1)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(2)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(3)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(4)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(5)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(6)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(7)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(8)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(9)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(10)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(11)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(12)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(13)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(14)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(15)
    TO_CONTIGUOUS_FORWARD_GPU_PARALLEL(16)
    default: break;
#undef TO_CONTIGUOUS_FORWARD_GPU_PARALLEL
  }
}

}  // namespace

template<typename T>
struct ToContiguousUtil<DeviceType::kCUDA, T> : ToContiguousUtilBase {
  using ToContiguousUtilBase::ToContiguousUtilBase;
  static constexpr size_t dsize = sizeof(T);
  void operator()() {
    int constant_memory_size = 0;
    const size_t ndims = contiguous_dim + 1;
    if (ndims == 0) {
      // 0-dim tensor
      OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, block_size * dsize, cudaMemcpyDeviceToDevice,
                                    stream->As<ep::CudaStream>()->cuda_stream()));
    } else {
      bool is_same = true;
      for (int64_t i = contiguous_dim; i != -1; --i) {
        if (out_stride[i] != in_stride[i]) {
          is_same = false;
          break;
        }
      }
      if (is_same) {
        // if input tensor's strides equals to output's, than just copy one memory-contiguous tensor
        OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, element_count * dsize,
                                      cudaMemcpyDeviceToDevice,
                                      stream->As<ep::CudaStream>()->cuda_stream()));
      } else {
        if (element_count < GetMaxVal<int32_t>()) {
          LaunchToContiguousKernel<T, int32_t>(stream, element_count, ndims, block_size, in_stride,
                                               out_stride, in_dptr, out_dptr);
        } else {
          LaunchToContiguousKernel<T, int64_t>(stream, element_count, ndims, block_size, in_stride,
                                               out_stride, in_dptr, out_dptr);
        }
      }
    }
  }
};

#define INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA(cpp_type, data_type) \
  template struct ToContiguousUtil<DeviceType::kCUDA, cpp_type>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA,
                     TO_CONTIGUOUS_COMMON_TYPES TO_CONTIGUOUS_CUDA_SPECIAL_TYPE)

}  // namespace oneflow
