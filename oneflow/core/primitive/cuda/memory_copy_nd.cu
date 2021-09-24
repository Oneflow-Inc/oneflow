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

#include "oneflow/core/primitive/include/memory_copy_nd.h"
#include "oneflow/core/stream/cuda_stream_context.h"
#include "oneflow/core/primitive/cuda/cuda_graph_support.h"
#include <cuda_runtime.h>
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace primitive {

namespace {

template<int32_t NDIMS, typename I>
struct SOA {
  I val[NDIMS];
};

template<int32_t NDIMS, typename T, typename I>
__global__ void CopyNDGpu(const int n, T* dst, const T* src,
                          NdIndexOffsetHelper<I, NDIMS> dst_helper,
                          NdIndexOffsetHelper<I, NDIMS> src_helper,
                          NdIndexOffsetHelper<I, NDIMS> copy_helper, SOA<NDIMS, I> dst_pos,
                          SOA<NDIMS, I> src_pos) {
  CUDA_1D_KERNEL_LOOP_T(I, i, n) {
    I copy_idx[NDIMS];
    I src_idx[NDIMS];
    I dst_idx[NDIMS];
    copy_helper.OffsetToNdIndex(i, copy_idx);
#pragma unroll
    for (I j = 0; j < NDIMS; j++) {
      src_idx[j] = src_pos.val[j] + copy_idx[j];
      dst_idx[j] = dst_pos.val[j] + copy_idx[j];
    }
    const I src_offset = src_helper.NdIndexToOffset(src_idx);
    const I dst_offset = dst_helper.NdIndexToOffset(dst_idx);
    dst[dst_offset] = src[src_offset];
  }
}

template<int32_t NDIMS>
size_t GetPackSize(void* dst, const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
                   const int64_t* src_dims, const int64_t* src_pos, const int64_t* extent) {
  const int64_t mask = dst_dims[NDIMS - 1] | src_dims[NDIMS - 1] | extent[NDIMS - 1]
                       | src_pos[NDIMS - 1] | dst_pos[NDIMS - 1]
                       | static_cast<int64_t>(reinterpret_cast<uintptr_t>(dst))
                       | static_cast<int64_t>(reinterpret_cast<uintptr_t>(src));
  if ((mask & 0xF) == 0) {
    return 16;
  } else if ((mask & 0x7) == 0) {
    return 8;
  } else if ((mask & 0x3) == 0) {
    return 4;
  } else if ((mask & 0x1) == 0) {
    return 2;
  } else {
    return 1;
  }
}

template<int32_t NDIMS, typename P, typename I>
void CopyNDByPackByIndexTypeGpu(cudaStream_t stream, void* dst, const int64_t* dst_dims,
                                const int64_t* dst_pos, const void* src, const int64_t* src_dims,
                                const int64_t* src_pos, const int64_t* extent) {
  constexpr size_t pack_size = sizeof(P);
  I dst_dim_arr[NDIMS];
  I src_dim_arr[NDIMS];
  I extent_dim_arr[NDIMS];
  SOA<NDIMS, I> src_pos_arr;
  SOA<NDIMS, I> dst_pos_arr;
  int copy_elem_cnt = 1;
  FOR_RANGE(int64_t, i, 0, NDIMS) {
    if (i == NDIMS - 1) {
      dst_pos_arr.val[i] = dst_pos[i] / pack_size;
      src_pos_arr.val[i] = src_pos[i] / pack_size;
      dst_dim_arr[i] = dst_dims[i] / pack_size;
      src_dim_arr[i] = src_dims[i] / pack_size;
      extent_dim_arr[i] = extent[i] / pack_size;
    } else {
      dst_pos_arr.val[i] = dst_pos[i];
      src_pos_arr.val[i] = src_pos[i];
      dst_dim_arr[i] = dst_dims[i];
      src_dim_arr[i] = src_dims[i];
      extent_dim_arr[i] = extent[i];
    }
    copy_elem_cnt *= extent_dim_arr[i];
  }
  NdIndexOffsetHelper<I, NDIMS> dst_helper(dst_dim_arr);
  NdIndexOffsetHelper<I, NDIMS> src_helper(src_dim_arr);
  NdIndexOffsetHelper<I, NDIMS> copy_helper(extent_dim_arr);

  CopyNDGpu<NDIMS, P, I>
      <<<BlocksNum4ThreadsNum(copy_elem_cnt), kCudaThreadsNumPerBlock, 0, stream>>>(
          copy_elem_cnt, reinterpret_cast<P*>(dst), reinterpret_cast<const P*>(src), dst_helper,
          src_helper, copy_helper, dst_pos_arr, src_pos_arr);
}

template<int32_t NDIMS, typename P>
void CopyNDByPackGpu(cudaStream_t stream, void* dst, const int64_t* dst_dims,
                     const int64_t* dst_pos, const void* src, const int64_t* src_dims,
                     const int64_t* src_pos, const int64_t* extent) {
  int64_t dst_elem_cnt = 1;
  int64_t src_elem_cnt = 1;
  FOR_RANGE(int64_t, i, 0, NDIMS) {
    dst_elem_cnt *= dst_dims[i];
    src_elem_cnt *= src_dims[i];
  }
  if (std::max(dst_elem_cnt, src_elem_cnt) > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
    CopyNDByPackByIndexTypeGpu<NDIMS, P, int64_t>(stream, dst, dst_dims, dst_pos, src, src_dims,
                                                  src_pos, extent);
  } else {
    CopyNDByPackByIndexTypeGpu<NDIMS, P, int32_t>(stream, dst, dst_dims, dst_pos, src, src_dims,
                                                  src_pos, extent);
  }
}

void GetDescInBytes(const size_t size_of_data_type, size_t num_dims, const int64_t* dst_dims,
                    const int64_t* dst_pos, const int64_t* src_dims, const int64_t* src_pos,
                    const int64_t* extent, int64_t* new_dst_dims, int64_t* new_dst_pos,
                    int64_t* new_src_dims, int64_t* new_src_pos, int64_t* new_extent) {
  FOR_RANGE(int64_t, i, 0, num_dims) {
    if (i == (num_dims - 1)) {
      new_dst_pos[i] = dst_pos[i] * size_of_data_type;
      new_src_pos[i] = src_pos[i] * size_of_data_type;
      new_dst_dims[i] = dst_dims[i] * size_of_data_type;
      new_src_dims[i] = src_dims[i] * size_of_data_type;
      new_extent[i] = extent[i] * size_of_data_type;
    } else {
      new_dst_pos[i] = dst_pos[i];
      new_src_pos[i] = src_pos[i];
      new_dst_dims[i] = dst_dims[i];
      new_src_dims[i] = src_dims[i];
      new_extent[i] = extent[i];
    }
  }
}

template<int32_t NDIMS>
void CopyNDGpuImpl(cudaStream_t stream, const size_t size_of_data_type, void* dst,
                   const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
                   const int64_t* src_dims, const int64_t* src_pos, const int64_t* extent) {
  LOG(ERROR) << "CopyNDGpuImpl " << NDIMS;
  int64_t new_dst_dims[NDIMS];
  int64_t new_dst_pos[NDIMS];
  int64_t new_src_dims[NDIMS];
  int64_t new_src_pos[NDIMS];
  int64_t new_extent[NDIMS];
  GetDescInBytes(size_of_data_type, NDIMS, dst_dims, dst_pos, src_dims, src_pos, extent,
                 new_dst_dims, new_dst_pos, new_src_dims, new_src_pos, new_extent);
  const size_t pack_size = GetPackSize<NDIMS>(dst, new_dst_dims, new_dst_pos, src, new_src_dims,
                                              new_src_pos, new_extent);
  if (pack_size == 1) {
    CopyNDByPackGpu<NDIMS, uint8_t>(stream, dst, new_dst_dims, new_dst_pos, src, new_src_dims,
                                    new_src_pos, new_extent);
  } else if (pack_size == 2) {
    CopyNDByPackGpu<NDIMS, uint16_t>(stream, dst, new_dst_dims, new_dst_pos, src, new_src_dims,
                                     new_src_pos, new_extent);
  } else if (pack_size == 4) {
    CopyNDByPackGpu<NDIMS, uint32_t>(stream, dst, new_dst_dims, new_dst_pos, src, new_src_dims,
                                     new_src_pos, new_extent);
  } else if (pack_size == 8) {
    CopyNDByPackGpu<NDIMS, uint64_t>(stream, dst, new_dst_dims, new_dst_pos, src, new_src_dims,
                                     new_src_pos, new_extent);
  } else if (pack_size == 16) {
    static_assert(sizeof(uint4) == 16, "");
    CopyNDByPackGpu<NDIMS, uint4>(stream, dst, new_dst_dims, new_dst_pos, src, new_src_dims,
                                  new_src_pos, new_extent);
  } else {
    UNIMPLEMENTED();
  }
}

void Copy1D(cudaStream_t stream, void* dst, const void* src, size_t count) {
  LOG(ERROR) << "copy 1D";
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
}

void DispatchLaunch(cudaStream_t stream, DataType data_type, size_t num_dims, void* dst,
                    const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
                    const int64_t* src_dims, const int64_t* src_pos, const int64_t* extent) {
  const size_t size_of_data_type = GetSizeOfDataType(data_type);
  if (num_dims == 0) {
    Copy1D(stream, (unsigned char*)dst, (unsigned char*)src, size_of_data_type);
  } else if (num_dims == 1) {
    Copy1D(stream, (unsigned char*)dst + dst_pos[0] * size_of_data_type,
           (unsigned char*)src + src_pos[0] * size_of_data_type, extent[0] * size_of_data_type);
  } else if (num_dims == 2) {
    CopyNDGpuImpl<2>(stream, size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos,
                     extent);
  } else if (num_dims == 3) {
    CopyNDGpuImpl<3>(stream, size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos,
                     extent);
  } else if (num_dims == 4) {
    CopyNDGpuImpl<4>(stream, size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos,
                     extent);
  } else if (num_dims == 5) {
    CopyNDGpuImpl<5>(stream, size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos,
                     extent);
  } else if (num_dims == 6) {
    CopyNDGpuImpl<6>(stream, size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos,
                     extent);
  } else {
    UNIMPLEMENTED();
  }
}

class MemoryCopyNdImpl : public MemoryCopyNd, public CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopyNdImpl);
  MemoryCopyNdImpl() = default;
  ~MemoryCopyNdImpl() override = default;

  void Launch(StreamContext* stream_ctx, DataType data_type, size_t num_dims, void* dst,
              const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
              const int64_t* src_dims, const int64_t* src_pos,
              const int64_t* extent) const override {
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    DispatchLaunch(cuda_stream, data_type, num_dims, dst, dst_dims, dst_pos, src, src_dims, src_pos,
                   extent);
  }
};

class MemoryCopyNdFactoryImpl : public MemoryCopyNdFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopyNdFactoryImpl);
  MemoryCopyNdFactoryImpl() = default;
  ~MemoryCopyNdFactoryImpl() override = default;

  std::unique_ptr<MemoryCopyNd> New() override {
    return std::unique_ptr<MemoryCopyNd>(new MemoryCopyNdImpl());
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, MemoryCopyNdFactory, MemoryCopyNdFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
