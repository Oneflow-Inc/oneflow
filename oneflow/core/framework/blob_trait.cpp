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
#include "oneflow/core/framework/blob_trait.h"

namespace oneflow {

namespace compatible_py {

std::shared_ptr<Shape> BlobHeaderTrait::static_shape() const { UNIMPLEMENTED(); }
std::shared_ptr<Shape> BlobHeaderTrait::shape() const { UNIMPLEMENTED(); }
std::shared_ptr<std::vector<std::shared_ptr<Shape>>> BlobHeaderTrait::shape_list() const {
  UNIMPLEMENTED();
}
DataType BlobHeaderTrait::dtype() const { UNIMPLEMENTED(); }
bool BlobHeaderTrait::is_tensor_list() const { UNIMPLEMENTED(); }

}  // namespace compatible_py

}  // namespace oneflow
