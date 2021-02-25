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
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

RtBlobDesc::RtBlobDesc(const BlobDesc& blob_desc) {
  BlobDescProto proto;
  blob_desc.ToProto(&proto);
  InitFromProto(proto);
}

RtBlobDesc::RtBlobDesc(const BlobDescProto& proto) { InitFromProto(proto); }

void RtBlobDesc::InitFromProto(const BlobDescProto& proto) {
  body_.InitFromProto(proto.body());
  header_.InitFromProto(proto.header());
  is_tensor_list_ = proto.is_tensor_list();
  is_dynamic_ = proto.is_dynamic();
  header_is_opaque_ = proto.header_is_opaque();
}

size_t RtBlobDesc::ByteSizeOfBlobHeader() const { return header_.ByteSize(); }

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return body_.ByteSize(); }

size_t RtBlobDesc::AlignedByteSizeOfBlobBody() const {
  return RoundUp(ByteSizeOfBlobBody(), BlobDesc::kAlignSize);
}

size_t RtBlobDesc::AlignedTotalByteSize() const {
  return ByteSizeOfBlobHeader() + AlignedByteSizeOfBlobBody();
}

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  return (body_ == rhs.body_) && (header_ == rhs.header_)
         && (is_tensor_list_ == rhs.is_tensor_list_) && (is_dynamic_ == rhs.is_dynamic_)
         && (header_is_opaque_ == rhs.header_is_opaque_);
}

}  // namespace oneflow
