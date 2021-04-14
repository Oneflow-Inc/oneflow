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
  shape_ = blob_desc.shape();
  data_type_ = blob_desc.data_type();
  is_dynamic_ = blob_desc.is_dynamic();
}

RtBlobDesc::RtBlobDesc(const BlobDescProto& proto) {
  shape_ = Shape(proto.shape());
  data_type_ = proto.data_type();
  is_dynamic_ = proto.is_dynamic();
}

size_t RtBlobDesc::ByteSizeOfBlobHeader() const { return shape_.NumAxes() * sizeof(int64_t); }

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return Capacity(); }

size_t RtBlobDesc::AlignedByteSizeOfBlobBody() const {
  return RoundUp(ByteSizeOfBlobBody(), BlobDesc::kAlignSize);
}

size_t RtBlobDesc::AlignedTotalByteSize() const {
  return ByteSizeOfBlobHeader() + AlignedByteSizeOfBlobBody();
}

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  return (shape_ == rhs.shape_) && (data_type_ == rhs.data_type_)
         && (is_dynamic_ == rhs.is_dynamic_);
}

}  // namespace oneflow
