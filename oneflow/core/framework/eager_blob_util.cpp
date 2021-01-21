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

EagerPhysicalBlobHeader::EagerPhysicalBlobHeader(
    const std::shared_ptr<Shape>& static_shape,
    const std::vector<std::shared_ptr<Shape>>& shape_list, DataType dtype, bool is_tensor_list)
    : static_shape_(static_shape),
      shape_list_(shape_list),
      dtype_(dtype),
      is_tensor_list_(is_tensor_list) {}
std::shared_ptr<Shape> EagerPhysicalBlobHeader::static_shape() const { return static_shape_; }
std::shared_ptr<Shape> EagerPhysicalBlobHeader::shape() const {
  CHECK_EQ(shape_list_.size(), 1);
  CHECK_EQ(is_tensor_list_, false);
  return shape_list_.at(0);
}
std::vector<std::shared_ptr<Shape>> EagerPhysicalBlobHeader::shape_list() const {
  CHECK_EQ(is_tensor_list_, true);
  return shape_list_;
}
DataType EagerPhysicalBlobHeader::dtype() const { return dtype_; }
bool EagerPhysicalBlobHeader::is_tensor_list() const { return is_tensor_list_; }

}  // namespace compatible_py

}  // namespace oneflow
