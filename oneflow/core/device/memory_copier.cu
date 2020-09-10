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
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

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

size_t GetPackSize(const MemoryCopyNdDesc& desc, const void* dst, const void* src) {
  const int64_t mask = desc.src_shape.dim_vec().back() | desc.dst_shape.dim_vec().back()
                       | desc.extent.dim_vec().back() | desc.src_pos.dim_vec().back()
                       | desc.dst_pos.dim_vec().back()
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

}  // namespace

template<int32_t NDIMS, typename P, typename I>
void CopyNDByPackByIndexTypeGpu(DeviceCtx* ctx, void* dst, const void* src,
                                const MemoryCopyNdDesc& desc) {
  CHECK_EQ(desc.dst_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.dst_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.extent.NumAxes(), NDIMS);
  constexpr size_t pack_size = sizeof(P);
  I dst_shape_dim_arr[NDIMS];
  I src_shape_dim_arr[NDIMS];
  I extent_dim_arr[NDIMS];
  SOA<NDIMS, I> src_pos;
  SOA<NDIMS, I> dst_pos;
  FOR_RANGE(int64_t, i, 0, NDIMS) {
    if (i == NDIMS - 1) {
      dst_pos.val[i] = desc.dst_pos.dim_vec().at(i) / pack_size;
      src_pos.val[i] = desc.src_pos.dim_vec().at(i) / pack_size;
      dst_shape_dim_arr[i] = desc.dst_shape.dim_vec().at(i) / pack_size;
      src_shape_dim_arr[i] = desc.src_shape.dim_vec().at(i) / pack_size;
      extent_dim_arr[i] = desc.extent.dim_vec().at(i) / pack_size;
    } else {
      dst_pos.val[i] = desc.dst_pos.dim_vec().at(i);
      src_pos.val[i] = desc.src_pos.dim_vec().at(i);
      dst_shape_dim_arr[i] = desc.dst_shape.dim_vec().at(i);
      src_shape_dim_arr[i] = desc.src_shape.dim_vec().at(i);
      extent_dim_arr[i] = desc.extent.dim_vec().at(i);
    }
  }
  NdIndexOffsetHelper<I, NDIMS> dst_helper(dst_shape_dim_arr);
  NdIndexOffsetHelper<I, NDIMS> src_helper(src_shape_dim_arr);
  NdIndexOffsetHelper<I, NDIMS> copy_helper(extent_dim_arr);
  const int64_t elem_cnt = desc.extent.elem_cnt() / pack_size;
  CopyNDGpu<NDIMS, P, I>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, reinterpret_cast<P*>(dst), reinterpret_cast<const P*>(src), dst_helper,
          src_helper, copy_helper, dst_pos, src_pos);
}

template<int32_t NDIMS, typename P>
void CopyNDByPackGpu(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  if (std::max(desc.dst_shape.elem_cnt(), desc.src_shape.elem_cnt())
      > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
    CopyNDByPackByIndexTypeGpu<NDIMS, P, int64_t>(ctx, dst, src, desc);
  } else {
    CopyNDByPackByIndexTypeGpu<NDIMS, P, int32_t>(ctx, dst, src, desc);
  }
}

template<int32_t NDIMS>
void CopyNDGpuImpl(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  const size_t pack_size = GetPackSize(desc, dst, src);
  if (pack_size == 1) {
    CopyNDByPackGpu<NDIMS, uint8_t>(ctx, dst, src, desc);
  } else if (pack_size == 2) {
    CopyNDByPackGpu<NDIMS, uint16_t>(ctx, dst, src, desc);
  } else if (pack_size == 4) {
    CopyNDByPackGpu<NDIMS, uint32_t>(ctx, dst, src, desc);
  } else if (pack_size == 8) {
    CopyNDByPackGpu<NDIMS, uint64_t>(ctx, dst, src, desc);
  } else if (pack_size == 16) {
    static_assert(sizeof(uint4) == 16, "");
    CopyNDByPackGpu<NDIMS, uint4>(ctx, dst, src, desc);
  } else {
    UNIMPLEMENTED();
  }
}

#define SPECIALIZE_COPY_ND_GPU_IMPL(NDIMS)                                        \
  template void CopyNDGpuImpl<NDIMS>(DeviceCtx * ctx, void* dst, const void* src, \
                                     const MemoryCopyNdDesc& desc);
SPECIALIZE_COPY_ND_GPU_IMPL(2)
SPECIALIZE_COPY_ND_GPU_IMPL(3)
SPECIALIZE_COPY_ND_GPU_IMPL(4)
SPECIALIZE_COPY_ND_GPU_IMPL(5)
SPECIALIZE_COPY_ND_GPU_IMPL(6)

}  // namespace oneflow
