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
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs) {
  return lhs.lbi() < rhs.lbi();
}

BlobDesc::BlobDesc(const std::shared_ptr<Shape>& shape, DataType dtype)
    : body_(shape, dtype), is_dynamic_(false) {}
BlobDesc::BlobDesc(const Shape& shape, DataType dtype)
    : BlobDesc(std::make_shared<Shape>(shape), dtype) {}

BlobDesc::BlobDesc(const BlobDescProto& proto) { InitFromProto(proto); }

BlobDesc::BlobDesc(const BlobDesc& other) { CopyFrom(other); }

void BlobDesc::InitFromProto(const BlobDescProto& proto) {
  body_.InitFromProto(proto.body());
  is_dynamic_ = proto.is_dynamic();
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  body_.ToProto(proto->mutable_body());
  proto->set_is_dynamic(is_dynamic_);

  StructPodDesc header;
  header.AddField(
      FieldKey::kTensorShape,
      TensorPodDesc(std::make_shared<Shape>(DimVector{shape().NumAxes()}), DataType::kInt64));
  header.ToProto(proto->mutable_header());
}

BlobDesc& BlobDesc::operator=(const BlobDesc& rhs) {
  this->CopyFrom(rhs);
  return *this;
}

void BlobDesc::CopyFrom(const BlobDesc& other) {
  *body_.mut_shape() = other.body_.shape();
  body_.set_data_type(other.body_.data_type());
  is_dynamic_ = other.is_dynamic_;
}

void BlobDesc::set_is_dynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return (body_ == rhs.body_) && (is_dynamic_ == rhs.is_dynamic_);
}

}  // namespace oneflow
