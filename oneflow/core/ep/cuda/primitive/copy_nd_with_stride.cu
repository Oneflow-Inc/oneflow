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

#include "oneflow/core/ep/include/primitive/copy_nd_with_stride.h"
#include "oneflow/core/ep/common/primitive/copy_nd_with_stride.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda_runtime.h>

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<size_t num_dims, size_t movement_size, typename IndexType>
__global__ void CopyNdWithStrideKernel(CopyNdWithStrideKernelParams<num_dims, IndexType> params) {
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
void LaunchKernel(Stream* stream, CopyNdWithStrideKernelParams<num_dims, IndexType> params) {
  cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
  CopyNdWithStrideKernel<num_dims, movement_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
}

class CopyNdWithStrideImpl : public CopyNdWithStride {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdWithStrideImpl);
  CopyNdWithStrideImpl() = default;
  ~CopyNdWithStrideImpl() override = default;

  void Launch(Stream* stream, DataType data_type, size_t num_dims, void* dst,
              const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
              const int64_t* src_dims, const int64_t* src_pos,
              const int64_t* extent, const int64_t* stride) const override {
    SimplifyThenLaunch(stream, data_type, num_dims, dst, dst_dims, dst_pos, src, src_dims, src_pos,
                       extent, stride);
  }
};

class CopyNdWithStrideFactoryImpl : public CopyNdWithStrideFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyNdWithStrideFactoryImpl);
  CopyNdWithStrideFactoryImpl() = default;
  ~CopyNdWithStrideFactoryImpl() override = default;

  std::unique_ptr<CopyNdWithStride> New(size_t max_num_dims) override {
    if (max_num_dims <= kMaxNumDims) {
      return std::unique_ptr<CopyNdWithStride>(new CopyNdWithStrideImpl());
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, CopyNdWithStrideFactory, CopyNdWithStrideFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
