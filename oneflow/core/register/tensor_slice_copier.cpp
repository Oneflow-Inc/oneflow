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
#include "oneflow/core/register/tensor_slice_copier.h"

namespace oneflow {

namespace {

TensorSliceView GetRawTenserSliceView(const TensorSliceView& view, DataType data_type) {
  const size_t size_of_data_type = GetSizeOfDataType(data_type);
  if (size_of_data_type == 1) {
    return view;
  } else {
    std::vector<Range> range_vec = view.range_vec();
    if (!view.IsEmpty()) {
      range_vec.back().mut_begin() = range_vec.back().begin() * size_of_data_type;
      range_vec.back().mut_end() = range_vec.back().end() * size_of_data_type;
    }
    return TensorSliceView(range_vec);
  }
}

}  // namespace

TensorSliceCopier::TensorSliceCopier(const TensorSliceView& dst_view,
                                     const TensorSliceView& src_view,
                                     const TensorSliceView& copy_view, const DataType data_type,
                                     const DeviceType device_type)
    : dst_view_(dst_view), src_view_(src_view), extent_(copy_view.shape()), data_type_(data_type) {
  copy_nd_primitive_ = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(
      device_type, dst_view_.shape().NumAxes());
  CHECK(dst_view.Contains(copy_view));
  CHECK(src_view.Contains(copy_view));
  dst_pos_ = copy_view.OffsetTo(dst_view);
  src_pos_ = copy_view.OffsetTo(src_view);
}

TensorSliceCopier::TensorSliceCopier(const TensorSliceView& dst_view,
                                     const TensorSliceView& src_view, const DataType data_type,
                                     const DeviceType device_type)
    : TensorSliceCopier(dst_view, src_view, dst_view.Intersect(src_view), data_type, device_type) {}

void TensorSliceCopier::Copy(ep::Stream* stream, void* dst, const void* src) const {
  copy_nd_primitive_->Launch(stream, data_type_, dst_view_.shape().NumAxes(), dst,
                             dst_view_.shape().dim_vec().data(), dst_pos_.dim_vec().data(), src,
                             src_view_.shape().dim_vec().data(), src_pos_.dim_vec().data(),
                             extent_.dim_vec().data());
}

void TensorSliceCopier::Copy(ep::Stream* stream, Blob* dst_blob, const Blob* src_blob) const {
  CHECK_EQ(dst_blob->data_type(), data_type_);
  CHECK_EQ(src_blob->data_type(), data_type_);
  CHECK_EQ(dst_view_.shape().elem_cnt(), dst_blob->shape().elem_cnt());
  CHECK_EQ(src_view_.shape().elem_cnt(), src_blob->shape().elem_cnt());
  Copy(stream, dst_blob->mut_dptr(), src_blob->dptr());
}

}  // namespace oneflow
