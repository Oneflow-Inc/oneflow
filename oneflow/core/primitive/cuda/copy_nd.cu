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

#include "oneflow/core/primitive/include/copy_nd.h"
#include "oneflow/core/primitive/common/copy_nd.h"
#include "oneflow/core/stream/cuda/cuda_stream_context.h"
#include <cuda_runtime.h>

namespace oneflow {

namespace primitive {

namespace {

template<size_t num_dims, size_t movement_size, typename IndexType>
__global__ void CopyNdKernel(CopyNdKernelParams<num_dims, IndexType> params) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);
  IndexType copy_index[num_dims];
  IndexType src_index[num_dims];
  IndexType dst_index[num_dims];
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, params.count) {
    params.copy_index_helper.OffsetToNdIndex(i, copy_index);
#pragma unroll
    for (size_t j = 0; j < num_dims; ++j) {
      src_index[j] = params.src_pos[j] + copy_index[j];
      dst_index[j] = params.dst_pos[j] + copy_index[j];
    }
    const IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
    const IndexType dst_offset = params.dst_index_helper.NdIndexToOffset(dst_index);
    dst[dst_offset] = src[src_offset];
  }
}

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(StreamContext* stream_ctx, CopyNdKernelParams<num_dims, IndexType> params) {
  cudaStream_t cuda_stream =
      CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
  CopyNdKernel<num_dims, movement_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
}

class CopyNdImpl : public CopyNd {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdImpl);
  CopyNdImpl() = default;
  ~CopyNdImpl() override = default;

  void Launch(StreamContext* stream_ctx, DataType data_type, size_t num_dims, void* dst,
              const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
              const int64_t* src_dims, const int64_t* src_pos,
              const int64_t* extent) const override {
    SimplifyThenLaunch(stream_ctx, data_type, num_dims, dst, dst_dims, dst_pos, src, src_dims,
                       src_pos, extent);
  }
};

class CopyNdFactoryImpl : public CopyNdFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdFactoryImpl);
  CopyNdFactoryImpl() = default;
  ~CopyNdFactoryImpl() override = default;

  std::unique_ptr<CopyNd> New(size_t max_num_dims) override {
    if (max_num_dims <= kMaxNumDims) {
      return std::unique_ptr<CopyNd>(new CopyNdImpl());
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, CopyNdFactory, CopyNdFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
