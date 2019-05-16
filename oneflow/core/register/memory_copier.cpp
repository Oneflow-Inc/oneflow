#include "oneflow/core/register/memory_copier.h"

namespace oneflow {

void BaseMemoryCopier::Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                              size_t src_pitch, size_t width, size_t height) const {
  char* dst_ptr = (char*)dst;
  const char* src_ptr = (const char*)src;
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

}  // namespace oneflow
