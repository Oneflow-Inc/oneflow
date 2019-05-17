#ifndef ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_COPIER_H_
#define ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_COPIER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/partial_tensor_view.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/memory_copier.h"

namespace oneflow {

class PartialTensorCopier final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PartialTensorCopier);
  PartialTensorCopier(const PartialTensorView& dst_view, const PartialTensorView& src_view,
                      const DataType data_type);
  virtual ~PartialTensorCopier() = default;

  void Exec(DeviceCtx* ctx, const MemoryCopier& copier, Blob* dst_blob, Blob* src_blob) const;

 private:
  mutable MemoryCopyNdDesc memory_copy_nd_desc_;
  const PartialTensorView dst_view_;
  const PartialTensorView src_view_;
  const DataType data_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_PARTIAL_TENSOR_VIEW_COPIER_H_
