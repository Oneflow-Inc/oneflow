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

 protected:
 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_COPIER_H_
