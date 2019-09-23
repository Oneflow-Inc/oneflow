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
}

size_t RtBlobDesc::ByteSizeOfBlobHeader() const { return header_.ByteSize(); }

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return body_.ByteSize(); }

size_t RtBlobDesc::AlignedByteSizeOfBlobBody(size_t align_size) const {
  return RoundUp(ByteSizeOfBlobBody(), align_size);
}

size_t AlignedTotalByteSize(size_t align_size) const {
  return ByteSizeOfBlobHeader() + AlignedByteSizeOfBlobBody(align_size);
}

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  return (body_ == rhs.body_) && (header_ == rhs.header_)
    && (num_of_lod_levels_ == rhs.num_of_lod_levels_)
    && (is_body_disabled_ == rhs.is_body_disabled_);
}

}  // namespace oneflow
