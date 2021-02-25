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
#ifndef ONEFLOW_CORE_FRAMEWORK_BLOB_TRAIT_H_
#define ONEFLOW_CORE_FRAMEWORK_BLOB_TRAIT_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {

namespace compatible_py {

class BlobHeaderTrait {
 public:
  BlobHeaderTrait() = default;
  ~BlobHeaderTrait() = default;

  virtual std::shared_ptr<Shape> static_shape() const;
  virtual std::shared_ptr<Shape> shape() const;
  virtual std::shared_ptr<std::vector<std::shared_ptr<Shape>>> shape_list() const;
  virtual DataType dtype() const;
  virtual bool is_tensor_list() const;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_BLOB_TRAIT_H_
