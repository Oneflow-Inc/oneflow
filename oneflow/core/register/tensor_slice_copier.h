#ifndef ONEFLOW_CORE_REGISTER_TENSOR_SLICE_COPIER_H_
#define ONEFLOW_CORE_REGISTER_TENSOR_SLICE_COPIER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/device/memory_copier.h"

namespace oneflow {

class TensorSliceCopier final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorSliceCopier);
  TensorSliceCopier(const TensorSliceView& dst_view, const TensorSliceView& src_view,
                    const TensorSliceView& copy_view, DataType data_type);
  TensorSliceCopier(const TensorSliceView& dst_view, const TensorSliceView& src_view,
                    DataType data_type);
  virtual ~TensorSliceCopier() = default;

  void Copy(DeviceCtx* ctx, const MemoryCopier& copier, void* dst, const void* src) const;
  void Copy(DeviceCtx* ctx, const MemoryCopier& copier, Blob* dst_blob, const Blob* src_blob) const;

 private:
  MemoryCopyNdDesc memory_copy_nd_desc_;
  const TensorSliceView dst_view_;
  const TensorSliceView src_view_;
  const DataType data_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_TENSOR_SLICE_COPIER_H_
