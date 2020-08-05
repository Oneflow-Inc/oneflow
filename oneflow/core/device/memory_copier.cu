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

template<int32_t NDIMS>
struct Int32Array {
  int32_t val[NDIMS];
};

template<int32_t NDIMS>
__global__ void CopyNDGpu(const int n, void* dst, const void* src,
                          NdIndexOffsetHelper<int64_t, NDIMS> dst_helper,
                          NdIndexOffsetHelper<int64_t, NDIMS> src_helper,
                          NdIndexOffsetHelper<int64_t, NDIMS> copy_helper,
                          Int32Array<NDIMS> dst_pos, Int32Array<NDIMS> src_pos) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t copy_idx[NDIMS];
    int64_t src_idx[NDIMS];
    int64_t dst_idx[NDIMS];
    copy_helper.OffsetToNdIndex(i, copy_idx);
#pragma unroll
    for (int64_t j = 0; j < NDIMS; j++) {
      src_idx[j] = src_pos.val[j] + copy_idx[j];
      dst_idx[j] = dst_pos.val[j] + copy_idx[j];
    }
    const int64_t src_offset = src_helper.NdIndexToOffset(src_idx);
    const int64_t dst_offset = dst_helper.NdIndexToOffset(dst_idx);
    unsigned char* dst_ptr = reinterpret_cast<unsigned char*>(dst) + dst_offset;
    const unsigned char* src_ptr = reinterpret_cast<const unsigned char*>(src) + src_offset;
    *dst_ptr = *src_ptr;
  }
}

}  // namespace

template<int32_t NDIMS>
void CopyNDGpuImpl(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  CHECK_EQ(desc.dst_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_pos.NumAxes(), NDIMS);
  CHECK_EQ(desc.dst_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.src_shape.NumAxes(), NDIMS);
  CHECK_EQ(desc.extent.NumAxes(), NDIMS);
  NdIndexOffsetHelper<int64_t, NDIMS> src_helper(desc.src_shape.dim_vec().data());
  NdIndexOffsetHelper<int64_t, NDIMS> dst_helper(desc.dst_shape.dim_vec().data());
  NdIndexOffsetHelper<int64_t, NDIMS> copy_helper(desc.extent.dim_vec().data());
  Int32Array<NDIMS> src_pos;
  Int32Array<NDIMS> dst_pos;
  FOR_RANGE(int64_t, i, 0, NDIMS) {
    dst_pos.val[i] = desc.dst_pos.At(i);
    src_pos.val[i] = desc.src_pos.At(i);
  }
  RUN_CUDA_KERNEL((CopyNDGpu<NDIMS>), ctx, desc.extent.elem_cnt(), desc.extent.elem_cnt(), dst, src,
                  dst_helper, src_helper, copy_helper, dst_pos, src_pos);
}

#define SPECIALIZE_COPY_ND_GPU_IMPL(NDIMS)                                        \
  template void CopyNDGpuImpl<NDIMS>(DeviceCtx * ctx, void* dst, const void* src, \
                                     const MemoryCopyNdDesc& desc);
SPECIALIZE_COPY_ND_GPU_IMPL(4)
SPECIALIZE_COPY_ND_GPU_IMPL(5)
SPECIALIZE_COPY_ND_GPU_IMPL(6)

}  // namespace oneflow
