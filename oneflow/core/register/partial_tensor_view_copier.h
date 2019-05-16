#ifndef ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_COPIER_H_
#define ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_COPIER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/partial_tensor_view.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/tensor_copy_desc.h"

namespace oneflow {

class PartialTensorViewCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PartialTensorViewCopier);
  PartialTensorViewCopier() = default;
  virtual ~PartialTensorViewCopier() = default;

  void CopyIntersection(DeviceCtx* ctx, const PartialTensorView& dst_view, Blob* dst_blob,
                        const PartialTensorView& src_view, const Blob* src_blob) const;
  virtual void DoMemcpyCopy(DeviceCtx* ctx, const TensorCopyDesc& desc) const;
  virtual void DoMemcpyCopy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const = 0;
  virtual void DoMemcpyCopy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                              size_t src_pitch, size_t width, size_t height) const;
  virtual void DoMemcpyCopy3D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                              size_t src_pitch, size_t width, size_t height) const;

 protected:
 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_COPIER_H_
