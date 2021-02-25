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
#include "oneflow/core/framework/object.h"
#include "oneflow/core/framework/blob_register.h"
#include "oneflow/core/framework/blob_trait.h"

namespace oneflow {

namespace compatible_py {

class EagerPhysicalBlobHeader : public BlobHeaderTrait {
 public:
  EagerPhysicalBlobHeader(const std::shared_ptr<Shape>& static_shape,
                          const std::shared_ptr<std::vector<std::shared_ptr<Shape>>>& shape_list,
                          DataType dtype, bool is_tensor_list);
  EagerPhysicalBlobHeader(const EagerPhysicalBlobHeader& other) = default;
  ~EagerPhysicalBlobHeader() = default;

  std::shared_ptr<Shape> static_shape() const override;
  std::shared_ptr<Shape> shape() const override;
  std::shared_ptr<std::vector<std::shared_ptr<Shape>>> shape_list() const override;
  DataType dtype() const override;
  bool is_tensor_list() const override;

 private:
  std::shared_ptr<Shape> static_shape_;
  std::shared_ptr<std::vector<std::shared_ptr<Shape>>> shape_list_;
  DataType dtype_;
  bool is_tensor_list_;
};

class EagerPhysicalBlob {
 public:
  EagerPhysicalBlob(
      const std::string& blob_name, const std::shared_ptr<BlobRegister>& blob_register,
      const std::function<std::shared_ptr<EagerPhysicalBlobHeader>(std::shared_ptr<BlobObject>)>&
          get_pysical_blob_header_cache);
  EagerPhysicalBlob(const EagerPhysicalBlob& other) = default;
  ~EagerPhysicalBlob();

  std::string logical_blob_name() const;
  std::string unique_name() const;
  std::shared_ptr<Shape> static_shape() const;
  std::shared_ptr<Shape> shape() const;
  DataType dtype() const;
  bool is_dynamic() const;
  bool is_tensor_list() const;
  std::shared_ptr<BlobObject> blob_object() const;
  std::string ToString() const;

 private:
  std::string blob_name_;
  std::shared_ptr<BlobObject> blob_object_;
  std::shared_ptr<BlobRegister> blob_register_;
  std::function<std::shared_ptr<EagerPhysicalBlobHeader>(std::shared_ptr<BlobObject>)>
      get_pysical_blob_header_cache_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_EAGER_BLOB_UTIL_H_
