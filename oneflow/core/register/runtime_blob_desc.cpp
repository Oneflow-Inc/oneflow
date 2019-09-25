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
  num_of_lod_levels_ = proto.num_of_lod_levels();
  is_body_disabled_ = proto.is_body_disabled();
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
         && (num_of_lod_levels_ == rhs.num_of_lod_levels_)
         && (is_body_disabled_ == rhs.is_body_disabled_) && (is_dynamic_ == rhs.is_dynamic_)
         && (header_is_opaque_ == rhs.header_is_opaque_);
}

}  // namespace oneflow
