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
#include "oneflow/core/framework/eager_blob_util.h"

namespace oneflow {

namespace compatible_py {

EagerPhysicalBlobHeader::EagerPhysicalBlobHeader(const std::shared_ptr<Shape>& static_shape,
                                                 const std::shared_ptr<Shape>& shape,
                                                 DataType dtype)
    : static_shape_(static_shape), shape_(shape), dtype_(dtype) {}
std::shared_ptr<Shape> EagerPhysicalBlobHeader::static_shape() const { return static_shape_; }

std::shared_ptr<Shape> EagerPhysicalBlobHeader::shape() const { return shape_; }

DataType EagerPhysicalBlobHeader::dtype() const { return dtype_; }

EagerPhysicalBlob::EagerPhysicalBlob(
    const std::string& blob_name, const std::shared_ptr<BlobRegister>& blob_register,
    const std::function<std::shared_ptr<EagerPhysicalBlobHeader>(std::shared_ptr<BlobObject>)>&
        get_pysical_blob_header_cache)
    : blob_name_(blob_name),
      blob_object_(blob_register->GetObject4BlobName(blob_name)),
      blob_register_(blob_register),
      get_pysical_blob_header_cache_(get_pysical_blob_header_cache) {}

EagerPhysicalBlob::~EagerPhysicalBlob() { blob_register_->ClearObject4BlobName(blob_name_); }

std::string EagerPhysicalBlob::logical_blob_name() const { return blob_name_; }

std::string EagerPhysicalBlob::unique_name() const { return blob_name_; }

std::shared_ptr<Shape> EagerPhysicalBlob::static_shape() const {
  return get_pysical_blob_header_cache_(blob_object_)->static_shape();
}

std::shared_ptr<Shape> EagerPhysicalBlob::shape() const {
  return get_pysical_blob_header_cache_(blob_object_)->shape();
}

DataType EagerPhysicalBlob::dtype() const {
  return get_pysical_blob_header_cache_(blob_object_)->dtype();
}

bool EagerPhysicalBlob::is_dynamic() const { return true; }

std::shared_ptr<BlobObject> EagerPhysicalBlob::blob_object() const { return blob_object_; }

std::string EagerPhysicalBlob::ToString() const {
  return std::string("EagerPhysicalBlob(shape=") + shape()->ToString()
         + ", dtype=" + DataType_Name(dtype());
}

}  // namespace compatible_py

}  // namespace oneflow
