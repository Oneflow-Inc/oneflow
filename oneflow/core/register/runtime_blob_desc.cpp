#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

RtBlobDesc::RtBlobDesc(const BlobDesc& blob_desc) {
  BlobDescProto blob_desc_proto;
  blob_desc.ToProto(&blob_desc_proto);
  InitFromProto(blob_desc_proto);
}

RtBlobDesc::RtBlobDesc(const BlobDescProto& blob_desc_proto) {
  InitFromProto(blob_desc_proto);
}

void RtBlobDesc::InitFromProto(const BlobDescProto& blob_desc_proto) {
  blob_desc_proto_ = blob_desc_proto;
  body_.InitFromProto(blob_desc_proto.body());
  header_.InitFromProto(blob_desc_proto.header());
}

size_t RtBlobDesc::RealByteSizeOfBlobHeader() const { return header_.ByteSize(); }

size_t RtBlobDesc::RealByteSizeOfBlobBody() const { return body_.ByteSize(); }

size_t RtBlobDesc::ByteSizeOfDataContentField() const { return body_desc_.ByteSize(); }

size_t RtBlobDesc::AlignedByteSizeOfBlobBody(size_t align_size) const {
  return RoundUp(RealByteSizeOfBlobBody(), align_size);
}

size_t AlignedTotalByteSize(size_t align_size) const {
  return RealByteSizeOfBlobHeader() + AlignedByteSizeOfBlobBody(align_size);
}

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  return blob_desc_proto_ == rhs.blob_desc_proto_;
}

}  // namespace oneflow
