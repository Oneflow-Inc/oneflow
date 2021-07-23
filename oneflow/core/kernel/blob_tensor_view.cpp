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
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace user_op {

BlobTensorView::BlobTensorView(Blob* blob) : blob_(blob) {}

const ShapeView& BlobTensorView::shape() const { return blob_->shape(); }

MutShapeView* BlobTensorView::mut_shape() { return blob_->mut_shape_view(); }

DataType BlobTensorView::data_type() const { return blob_->data_type(); }

const MemoryCase& BlobTensorView::mem_case() const { return blob_->mem_case(); }

const void* BlobTensorView::raw_dptr() const { return blob_->dptr(); }

void* BlobTensorView::mut_raw_dptr() { return blob_->mut_dptr(); }

void BlobTensorView::Reset(Blob* blob) { blob_ = blob; }

}  // namespace user_op

}  // namespace oneflow
