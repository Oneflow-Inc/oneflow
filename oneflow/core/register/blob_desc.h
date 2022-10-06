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
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class BlobDesc final {
 public:
  BlobDesc() = delete;
  ~BlobDesc() = default;

  // NOTE(chengcheng): Cannot using std::make_shared in header file, because it will cause
  //  Segmentation fault with unknown reason.
  BlobDesc(const Shape& shape, DataType dtype, bool is_dynamic);
  BlobDesc(const Shape& shape, const Stride& stride, DataType dtype, bool is_dynamic);
  BlobDesc(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype, bool is_dynamic);

  BlobDesc(const Shape& shape, DataType dtype);
  BlobDesc(const Shape& shape, const Stride& stride, DataType dtype);
  BlobDesc(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype);
  explicit BlobDesc(DataType dtype);
  explicit BlobDesc(const BlobDescProto& proto);
  explicit BlobDesc(const BlobDesc&);

  BlobDesc& operator=(const BlobDesc&);

  const Shape& shape() const {
    CHECK(shape_.operator bool());
    return *shape_;
  }
  const Stride& stride() const {
    CHECK(stride_.operator bool());
    return *stride_;
  }
  const std::shared_ptr<const Shape>& shape_ptr() const { return shape_.shared_from_symbol(); }
  const std::shared_ptr<const Stride>& stride_ptr() const { return stride_.shared_from_symbol(); }

  void set_shape(const Shape& shape) { this->shape_ = SymbolOf(shape); }
  void set_stride(const Stride& stride) { this->stride_ = SymbolOf(stride); }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType data_type) { data_type_ = data_type; }

  bool is_dynamic() const { return is_dynamic_; }
  void set_is_dynamic(bool is_dynamic);

  bool operator==(const BlobDesc&) const;
  void ToProto(BlobDescProto*) const;

  void CopyFrom(const BlobDesc&);

  size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t AlignedByteSizeOfBlobHeader() const;
  size_t AlignedByteSizeOfBlobBody() const;
  size_t AlignedTotalByteSize() const;

 private:
  Symbol<Shape> shape_;
  Symbol<Stride> stride_;
  DataType data_type_;
  bool is_dynamic_;
};

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
