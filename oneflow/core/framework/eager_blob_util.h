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
#ifndef ONEFLOW_CORE_FRAMEWORK_EAGER_BLOB_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_EAGER_BLOB_UTIL_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

namespace compatible_py {

class EagerPhysicalBlobHeader final {
 public:
  EagerPhysicalBlobHeader(const std::shared_ptr<Shape>& static_shape,
                          const std::vector<std::shared_ptr<Shape>>& shape_list, DataType dtype,
                          bool is_tensor_list);
  ~EagerPhysicalBlobHeader() = default;

  std::shared_ptr<Shape> static_shape() const;
  std::shared_ptr<Shape> shape() const;
  std::vector<std::shared_ptr<Shape>> shape_list() const;
  DataType dtype() const;
  bool is_tensor_list() const;

 private:
  std::shared_ptr<Shape> static_shape_;
  std::vector<std::shared_ptr<Shape>> shape_list_;
  DataType dtype_;
  bool is_tensor_list_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_EAGER_BLOB_UTIL_H_
