#ifndef ONEFLOW_CORE_REGISTER_MEMORY_COPIER_H_
#define ONEFLOW_CORE_REGISTER_MEMORY_COPIER_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/register/tensor_copy_desc.h"

namespace oneflow {

class MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopier);
  virtual ~MemoryCopier() = default;

  virtual void Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const = 0;
  virtual void Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                      size_t src_pitch, size_t width, size_t height) const = 0;
  virtual void Copy3D(DeviceCtx* ctx, const TensorCopyDesc& desc) const = 0;
  virtual void CopyND(DeviceCtx* ctx, const TensorCopyDesc& desc) const = 0;
};

class BaseMemoryCopier : virtual public MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BaseMemoryCopier);
  ~BaseMemoryCopier() override = default;

  void Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
              size_t width, size_t height) const override;
  void Copy3D(DeviceCtx* ctx, const TensorCopyDesc& desc) const override;
  void CopyND(DeviceCtx* ctx, const TensorCopyDesc& desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_MEMORY_COPIER_H_
