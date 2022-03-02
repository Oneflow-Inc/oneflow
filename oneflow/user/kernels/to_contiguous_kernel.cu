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
#include "oneflow/core/graph/task_node.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;
int num_threads() { return 32 * 4; }
int thread_work_size() { return 4; }
int block_work_size() { return thread_work_size() * num_threads(); }
int GetMinThreadNum(int64_t elem_cnt) { return num_threads(); }
int GetNumBlocks(int64_t elem_cnt) {
  return (elem_cnt + block_work_size() - 1) / block_work_size();
}

struct StrideParam {
  int32_t stride[SHAPE_MAX_AXIS_SIZE];

  // NOLINTNEXTLINE
  StrideParam(const int64_t* stride_vec, const size_t ndim) {
    for (size_t i = 0; i < ndim; ++i) { stride[i] = stride_vec[i]; }
  }
};

template<typename IndexType, int ndim>
__device__ __forceinline__ IndexType compute_index(IndexType out_offset,
                                                   const StrideParam& out_params,
                                                   const StrideParam& in_params) {
  IndexType in_offset = 0;
  IndexType remaining = out_offset;

#pragma unroll
  for (int i = 0; i < ndim; ++i) {
    const IndexType idx = static_cast<IndexType>(remaining / out_params.stride[i]);
    remaining -= idx * out_params.stride[i];
    in_offset += idx * in_params.stride[i];
  }
  return in_offset;
}

template<typename T, typename IndexType, int ndim>
__global__ void ToContiguousForwardGpuParallel(IndexType count, const StrideParam in_stride,
                                               const StrideParam out_stride, const T* in_dptr,
                                               T* out_dptr) {
  IndexType remaining = count - 512 * blockIdx.x;
  IndexType idx = blockIdx.x;
  IndexType thread_idx = threadIdx.x;
#pragma unroll
  for (IndexType i = 0; i < 4; i++) {
    if (thread_idx >= remaining) { return; }
    IndexType out_idx = thread_idx + 512 * idx;
    IndexType in_idx = compute_index<IndexType, ndim>(out_idx, out_stride, in_stride);
    out_dptr[out_idx] = in_dptr[in_idx];
    thread_idx += 128;
  }
}

template<typename T, typename IndexType, size_t pack_size>
void LaunchToContiguousKernel(ep::Stream* stream, IndexType count, const size_t ndim,
                              IndexType block_size, const std::vector<int64_t>& in_stride,
                              const StrideVector& out_stride, const char* in_dptr, char* out_dptr) {
  const int num_blocks = GetNumBlocks(count);
  const int num_threads = GetMinThreadNum(count);
  StrideParam param_in_stride(in_stride.data(), ndim), param_out_stride(out_stride.data(), ndim);

  switch (ndim) {
    case 1:
      ToContiguousForwardGpuParallel<T, IndexType, 1>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 2:
      ToContiguousForwardGpuParallel<T, IndexType, 2>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 3:
      ToContiguousForwardGpuParallel<T, IndexType, 3>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 4:
      ToContiguousForwardGpuParallel<T, IndexType, 4>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 5:
      ToContiguousForwardGpuParallel<T, IndexType, 5>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 6:
      ToContiguousForwardGpuParallel<T, IndexType, 6>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 7:
      ToContiguousForwardGpuParallel<T, IndexType, 7>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 8:
      ToContiguousForwardGpuParallel<T, IndexType, 8>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 9:
      ToContiguousForwardGpuParallel<T, IndexType, 9>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 10:
      ToContiguousForwardGpuParallel<T, IndexType, 10>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 11:
      ToContiguousForwardGpuParallel<T, IndexType, 11>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 12:
      ToContiguousForwardGpuParallel<T, IndexType, 12>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 13:
      ToContiguousForwardGpuParallel<T, IndexType, 13>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 14:
      ToContiguousForwardGpuParallel<T, IndexType, 14>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 15:
      ToContiguousForwardGpuParallel<T, IndexType, 15>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    case 16:
      ToContiguousForwardGpuParallel<T, IndexType, 16>
          <<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr),
              reinterpret_cast<T*>(out_dptr));
      break;
    default: break;
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
        constexpr size_t pack_size = cuda::elementwise::PackSize<T>();
        if (element_count < GetMaxVal<int32_t>()) {
          LaunchToContiguousKernel<T, int32_t, pack_size>(stream, element_count, ndims, block_size,
                                                          in_stride, out_stride, in_dptr, out_dptr);
        } else {
          LaunchToContiguousKernel<T, int64_t, pack_size>(stream, element_count, ndims, block_size,
                                                          in_stride, out_stride, in_dptr, out_dptr);
        }
      }
    }
  }
};

#define INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA(T) \
  template struct ToContiguousUtil<DeviceType::kCUDA, T>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA,
                     TO_CONTIGUOUS_TYPES TO_CONTIGUOUS_CUDA_SPECIAL_TYPE)

}  // namespace oneflow
