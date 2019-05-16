#include "oneflow/core/register/memory_copier.h"
#include <memory>

namespace oneflow {

void BaseMemoryCopier::Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                              size_t src_pitch, size_t width, size_t height) const {
  unsigned char* dst_ptr = (unsigned char*)dst;
  const unsigned char* src_ptr = (unsigned char*)src;
  FOR_RANGE(size_t, i, 0, height) {
    Copy1D(ctx, dst_ptr, src_ptr, width);
    dst_ptr += dst_pitch;
    src_ptr += src_pitch;
  }
}

void BaseMemoryCopier::Copy3D(DeviceCtx* ctx, const TensorCopyDesc& desc) const {
  CHECK_EQ(desc.NumAxes(), 3);
  const size_t dst_pitch = desc.dst_shape().Count(2);
  const size_t src_pitch = desc.src_shape().Count(2);
  const size_t width = desc.extent().At(2);
  const size_t height = desc.extent().At(1);
  const size_t depth = desc.extent().At(0);
  FOR_RANGE(size_t, i, 0, depth) {
    void* dst = (unsigned char*)desc.dst_ptr() + 0;
    const void* src = (unsigned char*)desc.src_ptr() + 0;
    Copy2D(ctx, dst, dst_pitch, src, src_pitch, width, height);
  }
}

void BaseMemoryCopier::CopyND(DeviceCtx* ctx, const TensorCopyDesc& desc) const { UNIMPLEMENTED(); }

void HostMemoryCopier::Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const {
  std::memcpy(dst, src, count);
}

#ifdef WITH_CUDA

void CudaMemoryCopier::Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const {
  CudaCheck(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, ctx->cuda_stream()));
}

void CudaMemoryCopier::Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                              size_t src_pitch, size_t width, size_t height) const {
  CudaCheck(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDefault,
                              ctx->cuda_stream()));
}

void CudaMemoryCopier::Copy3D(DeviceCtx* ctx, const TensorCopyDesc& desc) const {
  CHECK_EQ(desc.NumAxes(), 3);
  cudaMemcpy3DParms params{};
  params.srcPos = make_cudaPos(desc.src_pos().at(2), desc.src_pos().at(1), desc.src_pos().at(0));
  params.srcPtr = make_cudaPitchedPtr(const_cast<void*>(desc.src_ptr()), desc.src_shape().At(2),
                                      desc.src_shape().At(2), desc.src_shape().At(1));
  params.dstPos = make_cudaPos(desc.dst_pos().at(2), desc.dst_pos().at(1), desc.dst_pos().at(0));
  params.dstPtr = make_cudaPitchedPtr(desc.dst_ptr(), desc.dst_shape().At(2),
                                      desc.dst_shape().At(2), desc.dst_shape().At(1));
  params.extent = make_cudaExtent(desc.extent().At(2), desc.extent().At(1), desc.extent().At(0));
  params.kind = cudaMemcpyDefault;
  CudaCheck(cudaMemcpy3DAsync(&params, ctx->cuda_stream()));
}

#endif

}  // namespace oneflow
