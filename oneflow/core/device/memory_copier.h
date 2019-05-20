#ifndef ONEFLOW_CORE_DEVICE_MEMORY_COPIER_H_
#define ONEFLOW_CORE_DEVICE_MEMORY_COPIER_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/index.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

struct MemoryCopyNdDesc {
  void* dst_ptr;
  const void* src_ptr;
  Shape dst_shape;
  Shape src_shape;
  Index dst_pos;
  Index src_pos;
  Shape extent;

  MemoryCopyNdDesc CompressDims();
};

class MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopier);
  virtual ~MemoryCopier() = default;

  virtual void Copy(DeviceCtx* ctx, const MemoryCopyNdDesc& desc) const;

 protected:
  virtual void Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const = 0;
  virtual void Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                      size_t src_pitch, size_t width, size_t height) const = 0;
  virtual void Copy3D(DeviceCtx* ctx, const MemoryCopyNdDesc& desc) const = 0;
  virtual void CopyND(DeviceCtx* ctx, const MemoryCopyNdDesc& desc) const = 0;
};

class BaseMemoryCopier : virtual public MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BaseMemoryCopier);
  ~BaseMemoryCopier() override = default;

 protected:
  void Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
              size_t width, size_t height) const override;
  void Copy3D(DeviceCtx* ctx, const MemoryCopyNdDesc& desc) const override;
  void CopyND(DeviceCtx* ctx, const MemoryCopyNdDesc& desc) const override;
};

class HostMemoryCopier final : public BaseMemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostMemoryCopier);
  ~HostMemoryCopier() override = default;

 private:
  void Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const override;
};

#ifdef WITH_CUDA

class CudaMemoryCopier final : virtual public MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaMemoryCopier);
  ~CudaMemoryCopier() override = default;

 private:
  void Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const override;
  void Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
              size_t width, size_t height) const override;
  void Copy3D(DeviceCtx* ctx, const MemoryCopyNdDesc& desc) const override;
  void CopyND(DeviceCtx* ctx, const MemoryCopyNdDesc& desc) const override;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_MEMORY_COPIER_H_
