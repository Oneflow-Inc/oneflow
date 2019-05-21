#ifndef ONEFLOW_CORE_REGISTER_TENSOR_PARTIAL_COPIER_H_
#define ONEFLOW_CORE_REGISTER_TENSOR_PARTIAL_COPIER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/tensor_partial_view.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/device/memory_copier.h"

namespace oneflow {

class TensorPartialCopier final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorPartialCopier);
  TensorPartialCopier(const TensorPartialView& dst_view, const TensorPartialView& src_view,
                      DataType data_type);
  virtual ~TensorPartialCopier() = default;

  void Exec(DeviceCtx* ctx, const MemoryCopier& copier, Blob* dst_blob, const Blob* src_blob) const;

 private:
  mutable MemoryCopyNdDesc memory_copy_nd_desc_;
  const TensorPartialView dst_view_;
  const TensorPartialView src_view_;
  const DataType data_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_TENSOR_PARTIAL_COPIER_H_
