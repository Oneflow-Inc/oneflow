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

int64_t MemoryCopyNdDescGetNumAxes(const MemoryCopyNdDesc& desc) { return desc.extent.NumAxes(); }

void CheckPosExtent(const int64_t num_axes, const Shape& shape, const NdIndex& pos,
                    const Shape& extent) {
  CHECK_EQ(shape.NumAxes(), num_axes);
  CHECK_EQ(pos.NumAxes(), num_axes);
  CHECK_EQ(extent.NumAxes(), num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    CHECK_GE(pos.At(i), 0);
    CHECK_GT(extent.At(i), 0);
    CHECK_GT(shape.At(i), 0);
    CHECK_LE(pos.At(i) + extent.At(i), shape.At(i));
  }
}

void CheckMemoryCopyNdDesc(const MemoryCopyNdDesc& desc) {
  const int64_t num_axes = MemoryCopyNdDescGetNumAxes(desc);
  CHECK_GT(num_axes, 0);
  CheckPosExtent(num_axes, desc.dst_shape, desc.dst_pos, desc.extent);
  CheckPosExtent(num_axes, desc.src_shape, desc.src_pos, desc.extent);
}

template<typename T>
MemoryCopyNdDesc GetDescInBytes(const MemoryCopyNdDesc& desc) {
  MemoryCopyNdDesc desc_in_bytes;
  DimVector dst_shape_vec;
  DimVector src_shape_vec;
  DimVector dst_pos_vec;
  DimVector src_pos_vec;
  DimVector extent_vec;
  FOR_RANGE(int64_t, i, 0, MemoryCopyNdDescGetNumAxes(desc)) {
    if (i == (MemoryCopyNdDescGetNumAxes(desc) - 1)) {
      const int64_t size_of_data_type = sizeof(T);
      dst_shape_vec.push_back(desc.dst_shape.At(i) * size_of_data_type);
      src_shape_vec.push_back(desc.src_shape.At(i) * size_of_data_type);
      dst_pos_vec.push_back(desc.dst_pos.At(i) * size_of_data_type);
      src_pos_vec.push_back(desc.src_pos.At(i) * size_of_data_type);
      extent_vec.push_back(desc.extent.At(i) * size_of_data_type);
    } else {
      dst_shape_vec.push_back(desc.dst_shape.At(i));
      src_shape_vec.push_back(desc.src_shape.At(i));
      dst_pos_vec.push_back(desc.dst_pos.At(i));
      src_pos_vec.push_back(desc.src_pos.At(i));
      extent_vec.push_back(desc.extent.At(i));
    }
  }
  desc_in_bytes.dst_shape = Shape(dst_shape_vec);
  desc_in_bytes.src_shape = Shape(src_shape_vec);
  desc_in_bytes.dst_pos = NdIndex(dst_pos_vec);
  desc_in_bytes.src_pos = NdIndex(src_pos_vec);
  desc_in_bytes.extent = Shape(extent_vec);
  return desc_in_bytes;
}

}  // namespace

template<int32_t NDIMS>
void CopyNDCpuImpl(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) {
  NdIndexOffsetHelper<int64_t, NDIMS> src_helper(desc.src_shape.dim_vec().data());
  NdIndexOffsetHelper<int64_t, NDIMS> dst_helper(desc.dst_shape.dim_vec().data());
  NdIndexOffsetHelper<int64_t, NDIMS> copy_helper(desc.extent.dim_vec().data());
  FOR_RANGE(int64_t, i, 0, desc.extent.elem_cnt()) {
    int64_t copy_idx[NDIMS];
    int64_t src_idx[NDIMS];
    int64_t dst_idx[NDIMS];
    copy_helper.OffsetToNdIndex(i, copy_idx);
    FOR_RANGE(int64_t, j, 0, NDIMS) {
      src_idx[j] = desc.src_pos.At(j) + copy_idx[j];
      dst_idx[j] = desc.dst_pos.At(j) + copy_idx[j];
    }
    const int64_t src_offset = src_helper.NdIndexToOffset(src_idx);
    const int64_t dst_offset = dst_helper.NdIndexToOffset(dst_idx);
    unsigned char* dst_ptr = reinterpret_cast<unsigned char*>(dst) + dst_offset;
    const unsigned char* src_ptr = reinterpret_cast<const unsigned char*>(src) + src_offset;
    *dst_ptr = *src_ptr;
  }
}

MemoryCopyNdDesc MemoryCopyNdDesc::CreateDimReducedDesc() const {
  MemoryCopyNdDesc reduced;
  DimVector dst_shape_vec;
  DimVector src_shape_vec;
  DimVector dst_pos_vec;
  DimVector src_pos_vec;
  DimVector extent_vec;
  FOR_RANGE(int64_t, i, 0, MemoryCopyNdDescGetNumAxes(*this)) {
    if (dst_shape.At(i) == src_shape.At(i) && dst_shape.At(i) == extent.At(i) && dst_pos.At(i) == 0
        && src_pos.At(i) == 0 && i != 0) {
      dst_shape_vec.back() *= extent.At(i);
      src_shape_vec.back() *= extent.At(i);
      dst_pos_vec.back() *= extent.At(i);
      src_pos_vec.back() *= extent.At(i);
      extent_vec.back() *= extent.At(i);
    } else {
      dst_shape_vec.push_back(dst_shape.At(i));
      src_shape_vec.push_back(src_shape.At(i));
      dst_pos_vec.push_back(dst_pos.At(i));
      src_pos_vec.push_back(src_pos.At(i));
      extent_vec.push_back(extent.At(i));
    }
  }
  reduced.dst_shape = Shape(dst_shape_vec);
  reduced.src_shape = Shape(src_shape_vec);
  reduced.dst_pos = NdIndex(dst_pos_vec);
  reduced.src_pos = NdIndex(src_pos_vec);
  reduced.extent = Shape(extent_vec);
  return reduced;
}

void MemoryCopier::Copy(DeviceCtx* ctx, void* dst, const void* src,
                        const MemoryCopyNdDesc& desc) const {
  CheckMemoryCopyNdDesc(desc);
  const int64_t num_axes = MemoryCopyNdDescGetNumAxes(desc);
  if (num_axes == 1) {
    Copy1D(ctx, (unsigned char*)dst + desc.dst_pos.At(0), (unsigned char*)src + desc.src_pos.At(0),
           desc.extent.At(0));
  } else if (num_axes == 2) {
    const size_t dst_pitch = desc.dst_shape.At(1);
    const size_t src_pitch = desc.src_shape.At(1);
    const size_t width = desc.extent.At(1);
    const size_t height = desc.extent.At(0);
    void* dst_2d = (unsigned char*)dst + desc.dst_pos.At(0) * dst_pitch + desc.dst_pos.At(1);
    const void* src_2d =
        (const unsigned char*)src + desc.src_pos.At(0) * src_pitch + desc.src_pos.At(1);
    Copy2D(ctx, dst_2d, dst_pitch, src_2d, src_pitch, width, height);
  } else if (num_axes == 3) {
    Copy3D(ctx, dst, src, desc);
  } else {
    CopyND(ctx, dst, src, desc);
  }
}

template<typename T>
void MemoryCopier::CopyElem(DeviceCtx* ctx, void* dst, const void* src,
                            const MemoryCopyNdDesc& desc) const {
  MemoryCopyNdDesc desc_in_bytes = GetDescInBytes<T>(desc);
  Copy(ctx, dst, src, desc_in_bytes);
}

void MemoryCopier::Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                          size_t src_pitch, size_t width, size_t height) const {
  unsigned char* dst_ptr = (unsigned char*)dst;
  const unsigned char* src_ptr = (unsigned char*)src;
  FOR_RANGE(size_t, i, 0, height) {
    Copy1D(ctx, dst_ptr, src_ptr, width);
    dst_ptr += dst_pitch;
    src_ptr += src_pitch;
  }
}

void MemoryCopier::Copy3D(DeviceCtx* ctx, void* dst, const void* src,
                          const MemoryCopyNdDesc& desc) const {
  const size_t dst_pitch = desc.dst_shape.Count(2);
  const size_t src_pitch = desc.src_shape.Count(2);
  const size_t dst_inner_area = desc.dst_shape.Count(1);
  const size_t src_inner_area = desc.src_shape.Count(1);
  const size_t width = desc.extent.At(2);
  const size_t height = desc.extent.At(1);
  const size_t depth = desc.extent.At(0);
  FOR_RANGE(size_t, i, 0, depth) {
    void* dst_2d = (unsigned char*)dst + (desc.dst_pos.At(0) + i) * dst_inner_area
                   + desc.dst_pos.At(1) * dst_pitch + desc.dst_pos.At(2);
    const void* src_2d = (unsigned char*)src + (desc.src_pos.At(0) + i) * src_inner_area
                         + desc.src_pos.At(1) * src_pitch + desc.src_pos.At(2);
    Copy2D(ctx, dst_2d, dst_pitch, src_2d, src_pitch, width, height);
  }
}

void MemoryCopier::CopyND(DeviceCtx* ctx, void* dst, const void* src,
                          const MemoryCopyNdDesc& desc) const {
  UNIMPLEMENTED();
}

void HostMemoryCopier::Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const {
  memcpy(dst, src, count);
}

void HostMemoryCopier::CopyND(DeviceCtx* ctx, void* dst, const void* src,
                              const MemoryCopyNdDesc& desc) const {
  const int32_t num_axes = desc.src_shape.NumAxes();
  if (num_axes == 4) {
    CopyNDCpuImpl<4>(ctx, dst, src, desc);
  } else if (num_axes == 5) {
    CopyNDCpuImpl<5>(ctx, dst, src, desc);
  } else if (num_axes == 6) {
    CopyNDCpuImpl<6>(ctx, dst, src, desc);
  } else {
    UNIMPLEMENTED();
  }
}

#ifdef WITH_CUDA

namespace {

bool CanCurDevAccessPointer(const void* ptr) {
  int device_id;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  cudaPointerAttributes attributes;
  OF_CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr));
  return (attributes.type == cudaMemoryTypeDevice && attributes.device == device_id)
         || (attributes.type == cudaMemoryTypeHost);
}

}  // namespace

void CudaAsyncMemoryCopier::Copy(DeviceCtx* ctx, void* dst, const void* src,
                                 const MemoryCopyNdDesc& desc) const {
  CheckMemoryCopyNdDesc(desc);
  const int64_t num_axes = MemoryCopyNdDescGetNumAxes(desc);
  const bool use_nd_impl =
      CanCurDevAccessPointer(dst) && CanCurDevAccessPointer(src) && (num_axes != 1);
  if (use_nd_impl) {
    CopyND(ctx, dst, src, desc);
  } else {
    if (num_axes == 1) {
      Copy1D(ctx, (unsigned char*)dst + desc.dst_pos.At(0),
             (unsigned char*)src + desc.src_pos.At(0), desc.extent.At(0));
    } else if (num_axes == 2) {
      const size_t dst_pitch = desc.dst_shape.At(1);
      const size_t src_pitch = desc.src_shape.At(1);
      const size_t width = desc.extent.At(1);
      const size_t height = desc.extent.At(0);
      void* dst_2d = (unsigned char*)dst + desc.dst_pos.At(0) * dst_pitch + desc.dst_pos.At(1);
      const void* src_2d =
          (const unsigned char*)src + desc.src_pos.At(0) * src_pitch + desc.src_pos.At(1);
      Copy2D(ctx, dst_2d, dst_pitch, src_2d, src_pitch, width, height);
    } else if (num_axes == 3) {
      Copy3D(ctx, dst, src, desc);
    } else {
      UNIMPLEMENTED();
    }
  }
}

void CudaAsyncMemoryCopier::Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const {
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, ctx->cuda_stream()));
}

void CudaAsyncMemoryCopier::Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                                   size_t src_pitch, size_t width, size_t height) const {
  OF_CUDA_CHECK(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDefault,
                                  ctx->cuda_stream()));
}

void CudaAsyncMemoryCopier::Copy3D(DeviceCtx* ctx, void* dst, const void* src,
                                   const MemoryCopyNdDesc& desc) const {
  cudaMemcpy3DParms params{};
  params.srcPos = make_cudaPos(desc.src_pos.At(2), desc.src_pos.At(1), desc.src_pos.At(0));
  params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), desc.src_shape.At(2),
                                      desc.src_shape.At(2), desc.src_shape.At(1));
  params.dstPos = make_cudaPos(desc.dst_pos.At(2), desc.dst_pos.At(1), desc.dst_pos.At(0));
  params.dstPtr =
      make_cudaPitchedPtr(dst, desc.dst_shape.At(2), desc.dst_shape.At(2), desc.dst_shape.At(1));
  params.extent = make_cudaExtent(desc.extent.At(2), desc.extent.At(1), desc.extent.At(0));
  params.kind = cudaMemcpyDefault;
  OF_CUDA_CHECK(cudaMemcpy3DAsync(&params, ctx->cuda_stream()));
}

void CudaAsyncMemoryCopier::CopyND(DeviceCtx* ctx, void* dst, const void* src,
                                   const MemoryCopyNdDesc& desc) const {
  const int32_t num_axes = desc.src_shape.NumAxes();
  if (num_axes == 2) {
    CopyNDGpuImpl<2>(ctx, dst, src, desc);
  } else if (num_axes == 3) {
    CopyNDGpuImpl<3>(ctx, dst, src, desc);
  } else if (num_axes == 4) {
    CopyNDGpuImpl<4>(ctx, dst, src, desc);
  } else if (num_axes == 5) {
    CopyNDGpuImpl<5>(ctx, dst, src, desc);
  } else if (num_axes == 6) {
    CopyNDGpuImpl<6>(ctx, dst, src, desc);
  } else {
    UNIMPLEMENTED();
  }
}
#endif

#ifdef WITH_HIP

namespace {

bool CanCurDevAccessPointer(const void* ptr) {
  int device_id;
  OF_HIP_CHECK(hipGetDevice(&device_id));
  hipPointerAttribute_t attributes;
  OF_HIP_CHECK(hipPointerGetAttributes(&attributes, ptr));
  return (attributes.memoryType == hipMemoryTypeDevice && attributes.device == device_id)
         || (attributes.memoryType == hipMemoryTypeHost);
}

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
  HIP_1D_KERNEL_LOOP_T(I, i, n) {
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
      <<<BlocksNum4ThreadsNum(elem_cnt), kHipThreadsNumPerBlock, 0, ctx->rocm_stream()>>>(
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

void CudaAsyncMemoryCopier::Copy(DeviceCtx* ctx, void* dst, const void* src,
                                 const MemoryCopyNdDesc& desc) const {
  CheckMemoryCopyNdDesc(desc);
  const int64_t num_axes = MemoryCopyNdDescGetNumAxes(desc);
  const bool use_nd_impl =
      CanCurDevAccessPointer(dst) && CanCurDevAccessPointer(src) && (num_axes != 1);
  if (use_nd_impl) {
    CopyND(ctx, dst, src, desc);
  } else {
    if (num_axes == 1) {
      Copy1D(ctx, (unsigned char*)dst + desc.dst_pos.At(0),
             (unsigned char*)src + desc.src_pos.At(0), desc.extent.At(0));
    } else if (num_axes == 2) {
      const size_t dst_pitch = desc.dst_shape.At(1);
      const size_t src_pitch = desc.src_shape.At(1);
      const size_t width = desc.extent.At(1);
      const size_t height = desc.extent.At(0);
      void* dst_2d = (unsigned char*)dst + desc.dst_pos.At(0) * dst_pitch + desc.dst_pos.At(1);
      const void* src_2d =
          (const unsigned char*)src + desc.src_pos.At(0) * src_pitch + desc.src_pos.At(1);
      Copy2D(ctx, dst_2d, dst_pitch, src_2d, src_pitch, width, height);
    } else if (num_axes == 3) {
      Copy3D(ctx, dst, src, desc);
    } else {
      UNIMPLEMENTED();
    }
  }
}

void CudaAsyncMemoryCopier::Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const {
  OF_HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyDefault, ctx->rocm_stream()));
}

void CudaAsyncMemoryCopier::Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                                   size_t src_pitch, size_t width, size_t height) const {
  OF_HIP_CHECK(hipMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, hipMemcpyDefault,
                                  ctx->rocm_stream()));
}

void CudaAsyncMemoryCopier::Copy3D(DeviceCtx* ctx, void* dst, const void* src,
                                   const MemoryCopyNdDesc& desc) const {
  hipMemcpy3DParms params{};
  params.srcPos = make_hipPos(desc.src_pos.At(2), desc.src_pos.At(1), desc.src_pos.At(0));
  params.srcPtr = make_hipPitchedPtr(const_cast<void*>(src), desc.src_shape.At(2),
                                      desc.src_shape.At(2), desc.src_shape.At(1));
  params.dstPos = make_hipPos(desc.dst_pos.At(2), desc.dst_pos.At(1), desc.dst_pos.At(0));
  params.dstPtr =
      make_hipPitchedPtr(dst, desc.dst_shape.At(2), desc.dst_shape.At(2), desc.dst_shape.At(1));
  params.extent = make_hipExtent(desc.extent.At(2), desc.extent.At(1), desc.extent.At(0));
  params.kind = hipMemcpyDefault;
  OF_HIP_CHECK(hipMemcpy3DAsync(&params, ctx->rocm_stream()));
}

void CudaAsyncMemoryCopier::CopyND(DeviceCtx* ctx, void* dst, const void* src,
                                   const MemoryCopyNdDesc& desc) const {
  const int32_t num_axes = desc.src_shape.NumAxes();
  if (num_axes == 2) {
    CopyNDGpuImpl<2>(ctx, dst, src, desc);
  } else if (num_axes == 3) {
    CopyNDGpuImpl<3>(ctx, dst, src, desc);
  } else if (num_axes == 4) {
    CopyNDGpuImpl<4>(ctx, dst, src, desc);
  } else if (num_axes == 5) {
    CopyNDGpuImpl<5>(ctx, dst, src, desc);
  } else if (num_axes == 6) {
    CopyNDGpuImpl<6>(ctx, dst, src, desc);
  } else {
    UNIMPLEMENTED();
  }
}
#endif

REGISTER_DEFAULT_MEMORY_COPIER(DeviceType::kCPU, []() { return new HostMemoryCopier(); });
#if defined(WITH_CUDA) || defined(WITH_HIP)
REGISTER_DEFAULT_MEMORY_COPIER(DeviceType::kGPU, []() { return new CudaAsyncMemoryCopier(); });
#endif

MemoryCopier* NewDefaultMemoryCopier(DeviceType device_type) {
  return std::unique_ptr<DefaultMemoryCopierCreator>(
             NewObj<int32_t, DefaultMemoryCopierCreator>(device_type))
      ->Create();
}

#define SPECIALIZE_COPY_ELEM(dtype)                                                        \
  template void MemoryCopier::CopyElem<dtype>(DeviceCtx * ctx, void* dst, const void* src, \
                                              const MemoryCopyNdDesc& desc) const;
SPECIALIZE_COPY_ELEM(float16)
SPECIALIZE_COPY_ELEM(float)
SPECIALIZE_COPY_ELEM(double)
SPECIALIZE_COPY_ELEM(int32_t)
SPECIALIZE_COPY_ELEM(int64_t)
SPECIALIZE_COPY_ELEM(int8_t)

#define SPECIALIZE_COPY_ND_CPU_IMPL(NDIMS)                                        \
  template void CopyNDCpuImpl<NDIMS>(DeviceCtx * ctx, void* dst, const void* src, \
                                     const MemoryCopyNdDesc& desc);
SPECIALIZE_COPY_ND_CPU_IMPL(4)
SPECIALIZE_COPY_ND_CPU_IMPL(5)
SPECIALIZE_COPY_ND_CPU_IMPL(6)

}  // namespace oneflow
