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
#include "oneflow/core/register/tensor_view.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

int64_t NumAxes4Blob(const Blob* blob) { return blob->static_shape().NumAxes(); }

const MemoryCase& MemCase4Blob(const Blob* blob) { return blob->mem_case(); }
DataType DataType4Blob(const Blob* blob) { return blob->data_type(); }

FullyMutTensorView::FullyMutTensorView(const Blob* blob, int64_t* shape_ptr, char* dptr)
    : MutTensorView<MutShapeView>(blob, shape_ptr, dptr) {}

void FullyMutTensorView::set_shape(const Shape& shape) {
  CheckCapacity(shape.elem_cnt());
  shape_view_ptr()->set_shape(shape);
}

void FullyMutTensorView::set_shape(const ShapeView& shape) {
  CheckCapacity(shape.elem_cnt());
  shape_view_ptr()->set_shape(shape);
}

void FullyMutTensorView::CheckCapacity(size_t shape_elem_cnt) const {
  size_t data_offset = dptr<char>() - blob()->dptr<char>();
  size_t capacity = blob()->blob_desc().ByteSizeOfBlobBody() - data_offset;
  CHECK_LE(shape_elem_cnt * GetSizeOfDataType(data_type()), capacity);
}

}  // namespace oneflow
