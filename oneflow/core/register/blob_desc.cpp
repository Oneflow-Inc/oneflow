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

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc) {
  // TODO(niuchong) : remove PackedBlob
  int64_t body_byte_size = 0;
  StructPodDesc opaque_header_pod_desc;
  std::unique_ptr<BlobDesc> ret;
  for (const auto& pair : lbi2blob_desc) {
    if (lbi2blob_desc.size() == 1) {
      ret.reset(new BlobDesc(*(pair.second)));
      break;
    }
    RtBlobDesc rt_blob_desc(*(pair.second));
    // CHECK(!rt_blob_desc.is_dynamic());
    body_byte_size += rt_blob_desc.AlignedByteSizeOfBlobBody();
    *opaque_header_pod_desc.MutStructField(NewFieldId(pair.first)) = rt_blob_desc.header_pod_desc();
  }
  if (lbi2blob_desc.size() > 1) {
    ret.reset(new BlobDesc(Shape(DimVector{body_byte_size}), DataType::kChar));
    ret->SetOpaqueHeader(opaque_header_pod_desc);
  }
  return ret;
}

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs) {
  return lhs.lbi() < rhs.lbi();
}

BlobDesc::BlobDesc(const Shape& shape, DataType dtype)
    : body_(shape, dtype), is_tensor_list_(false), is_dynamic_(false), opaque_header_() {}

BlobDesc::BlobDesc(const BlobDescProto& proto) { InitFromProto(proto); }

BlobDesc::BlobDesc(const BlobDesc& other) {
  // *body_.mut_shape() = other.body_.shape();
  // body_.set_data_type(other.body_.data_type());
  // header_ = other.header_;
  // is_tensor_list_ = other.is_tensor_list_;
  BlobDescProto proto;
  other.ToProto(&proto);
  InitFromProto(proto);
}

void BlobDesc::InitFromProto(const BlobDescProto& proto) {
  body_.InitFromProto(proto.body());
  is_tensor_list_ = proto.is_tensor_list();
  is_dynamic_ = proto.is_dynamic();
  if (proto.header_is_opaque()) {
    opaque_header_.reset(new StructPodDesc(proto.header()));
  } else {
    opaque_header_.reset(nullptr);
  }
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  body_.ToProto(proto->mutable_body());
  proto->set_is_tensor_list(is_tensor_list_);
  proto->set_is_dynamic(is_dynamic_);

  if (opaque_header_) {
    opaque_header_->ToProto(proto->mutable_header());
    proto->set_header_is_opaque(true);
  } else {
    StructPodDesc header;
    int64_t shape_num_axes = shape().NumAxes();
    header.AddField(FieldKey::kTensorListLength,
                    TensorPodDesc(Shape(DimVector{1LL}), DataType::kInt64));
    header.AddField(FieldKey::kTensorListSlicesLength,
                    TensorPodDesc(Shape(DimVector{1LL}), DataType::kInt64));
    header.AddField(FieldKey::kLastTensorDataOffset,
                    TensorPodDesc(Shape(DimVector{1LL}), DataType::kInt64));
    int64_t shape_list_size = 1;
    if (is_tensor_list_ && shape().NumAxes() > 0) {
      int32_t batch_axis = 0;  // TODO: batch_axis isn't always 0
      shape_list_size = shape().At(batch_axis);
    }
    header.AddField(
        FieldKey::kTensorShapeList,
        TensorPodDesc(Shape(DimVector{shape_list_size * shape_num_axes}), DataType::kInt64));
    header.AddField(FieldKey::kTensorListSlices,
                    TensorPodDesc(Shape(DimVector{shape_list_size}), DataType::kInt64));
    header.ToProto(proto->mutable_header());
    proto->set_header_is_opaque(false);
  }
}

BlobDesc& BlobDesc::operator=(const BlobDesc& rhs) {
  this->CopyFrom(rhs);
  return *this;
}

void BlobDesc::CopyFrom(const BlobDesc& other) {
  BlobDescProto proto;
  other.ToProto(&proto);
  this->InitFromProto(proto);
}

void BlobDesc::SetOpaqueHeader(const StructPodDesc& header_pod_desc) {
  CHECK(!is_dynamic_);
  CHECK_EQ(is_tensor_list_, false);
  CHECK_GT(header_pod_desc.ByteSize(), 0);
  opaque_header_.reset(new StructPodDesc(header_pod_desc));
}

void BlobDesc::set_is_dynamic(bool is_dynamic) {
  if (!is_dynamic) { CHECK_EQ(false, is_tensor_list_); }
  is_dynamic_ = is_dynamic;
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return (body_ == rhs.body_) && (is_tensor_list_ == rhs.is_tensor_list_)
         && (is_dynamic_ == rhs.is_dynamic_) && (opaque_header_ == rhs.opaque_header_);
}

}  // namespace oneflow
