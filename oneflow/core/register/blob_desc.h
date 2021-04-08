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
#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include <memory>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class BlobDesc final {
 public:
  BlobDesc() = delete;
  ~BlobDesc() = default;

  BlobDesc(const Shape& shape, DataType dtype, bool is_dynamic)
      : shape_(std::make_shared<Shape>(shape)), data_type_(dtype), is_dynamic_(is_dynamic) {}
  BlobDesc(const std::shared_ptr<Shape>& shape, DataType dtype, bool is_dynamic)
      : shape_(shape), data_type_(dtype), is_dynamic_(is_dynamic) {}

  BlobDesc(const Shape& shape, DataType dtype) : BlobDesc(shape, dtype, false) {}
  BlobDesc(const std::shared_ptr<Shape>& shape, DataType dtype) : BlobDesc(shape, dtype, false) {}
  explicit BlobDesc(DataType dtype) : BlobDesc(Shape(), dtype, false) {}
  explicit BlobDesc(const BlobDescProto& proto);
  explicit BlobDesc(const BlobDesc&);

  static const int32_t kAlignSize = 512;

  BlobDesc& operator=(const BlobDesc&);

  const Shape& shape() const { return *CHECK_NOTNULL(shape_.get()); }
  Shape& mut_shape() { return *CHECK_NOTNULL(shape_.get()); }
  void set_shape(const Shape& shape) { *CHECK_NOTNULL(shape_.get()) = shape; }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType val) { data_type_ = val; }

  bool is_dynamic() const { return is_dynamic_; }
  void set_is_dynamic(bool);

  bool operator==(const BlobDesc&) const;
  void ToProto(BlobDescProto*) const;

  void CopyFrom(const BlobDesc&);

 private:
  std::shared_ptr<Shape> shape_;
  DataType data_type_;
  bool is_dynamic_;
};

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
