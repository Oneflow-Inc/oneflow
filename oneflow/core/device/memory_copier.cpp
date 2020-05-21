#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/common/auto_registration_factory.h"

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

}  // namespace

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

#ifdef WITH_CUDA

void CudaAsyncMemoryCopier::Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const {
  CudaCheck(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, ctx->cuda_stream()));
}

void CudaAsyncMemoryCopier::Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                                   size_t src_pitch, size_t width, size_t height) const {
  CudaCheck(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDefault,
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
  CudaCheck(cudaMemcpy3DAsync(&params, ctx->cuda_stream()));
}

void CudaAsyncMemoryCopier::CopyND(DeviceCtx* ctx, void* dst, const void* src,
                                   const MemoryCopyNdDesc& desc) const {
  UNIMPLEMENTED();
}

REGISTER_DEFAULT_MEMORY_COPIER(DeviceType::kCPU, []() { return new HostMemoryCopier(); });

#ifdef WITH_CUDA

REGISTER_DEFAULT_MEMORY_COPIER(DeviceType::kGPU, []() { return new CudaAsyncMemoryCopier(); });

#endif

MemoryCopier* NewDefaultMemoryCopier(DeviceType device_type) {
  return std::unique_ptr<DefaultMemoryCopierCreator>(
             NewObj<DefaultMemoryCopierCreator>(device_type))
      ->Create();
}

#endif

}  // namespace oneflow
