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
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs) {
  return lhs.lbi() < rhs.lbi();
}

BlobDesc::BlobDesc(const Shape& shape, DataType dtype, bool is_dynamic)
    : shape_(SymbolOf(shape)),
      stride_(SymbolOf(Stride(shape))),
      data_type_(dtype),
      is_dynamic_(is_dynamic) {}
BlobDesc::BlobDesc(const Shape& shape, const Stride& stride, DataType dtype, bool is_dynamic)
    : shape_(SymbolOf(shape)),
      stride_(SymbolOf(stride)),
      data_type_(dtype),
      is_dynamic_(is_dynamic) {}
BlobDesc::BlobDesc(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype, bool is_dynamic)
    : shape_(shape), stride_(stride), data_type_(dtype), is_dynamic_(is_dynamic) {}
BlobDesc::BlobDesc(const Shape& shape, DataType dtype)
    : BlobDesc(shape, Stride(shape), dtype, false) {}
BlobDesc::BlobDesc(const Shape& shape, const Stride& stride, DataType dtype)
    : BlobDesc(shape, stride, dtype, false) {}
BlobDesc::BlobDesc(Symbol<Shape> shape, Symbol<Stride> stride, DataType dtype)
    : BlobDesc(shape, stride, dtype, false) {}
BlobDesc::BlobDesc(DataType dtype) : BlobDesc(Shape(), Stride(), dtype, false) {}

BlobDesc::BlobDesc(const BlobDescProto& proto) {
  shape_ = SymbolOf(Shape(proto.shape()));
  stride_ = SymbolOf(Stride(proto.stride()));
  data_type_ = proto.data_type();
  is_dynamic_ = proto.is_dynamic();
}

BlobDesc::BlobDesc(const BlobDesc& other)
    : shape_(other.shape_),
      stride_(other.stride_),
      data_type_(other.data_type()),
      is_dynamic_(other.is_dynamic()) {}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  shape().ToProto(proto->mutable_shape());
  stride().ToProto(proto->mutable_stride());
  proto->set_data_type(data_type_);
  proto->set_is_dynamic(is_dynamic_);
}

BlobDesc& BlobDesc::operator=(const BlobDesc& rhs) {
  this->CopyFrom(rhs);
  return *this;
}

void BlobDesc::CopyFrom(const BlobDesc& other) {
  set_shape(other.shape());
  set_stride(other.stride());
  set_data_type(other.data_type());
  set_is_dynamic(other.is_dynamic());
}

void BlobDesc::set_is_dynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return (shape() == rhs.shape()) && (stride() == rhs.stride()) && (data_type() == rhs.data_type())
         && (is_dynamic() == rhs.is_dynamic());
}

size_t BlobDesc::ByteSizeOfBlobHeader() const {
  return shape().is_initialized() ? shape().NumAxes() * sizeof(int64_t) : 0;
}

size_t BlobDesc::AlignedByteSizeOfBlobHeader() const {
  return shape().is_initialized()
             ? RoundUp(shape().NumAxes() * sizeof(int64_t), kBlobHeaderAlignSize)
             : RoundUp(0, kBlobHeaderAlignSize);
}

size_t BlobDesc::ByteSizeOfBlobBody() const {
  return shape().is_initialized() ? shape().elem_cnt() * GetSizeOfDataType(data_type()) : 0;
}

size_t BlobDesc::AlignedByteSizeOfBlobBody() const {
  return RoundUp(ByteSizeOfBlobBody(), kBlobBodyAlignSize);
}

size_t BlobDesc::AlignedTotalByteSize() const {
  return AlignedByteSizeOfBlobHeader() + AlignedByteSizeOfBlobBody();
}

}  // namespace oneflow
