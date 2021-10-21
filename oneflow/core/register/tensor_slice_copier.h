/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
