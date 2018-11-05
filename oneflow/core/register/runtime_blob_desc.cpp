#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

RtBlobDesc::RtBlobDesc(const BlobDesc& blob_desc) {
  BlobDescProto blob_desc_proto;
  blob_desc.ToProto(&blob_desc_proto);
  InitFromProto(blob_desc_proto);
}

RtBlobDesc::RtBlobDesc(const BlobDescProto& blob_desc_proto) { InitFromProto(blob_desc_proto); }

void RtBlobDesc::InitFromProto(const BlobDescProto& blob_desc_proto) {
  blob_desc_proto_ = blob_desc_proto;
  body_desc_ = FieldDesc(blob_desc_proto.body());
  header_pod_desc_.InitFromProto(blob_desc_proto.header().header_pod_desc());
  if (blob_desc_proto.has_dim0_inner_shape()) {
    dim0_inner_shape_.reset(new Shape(blob_desc_proto.dim0_inner_shape()));
  }
}

const Shape& RtBlobDesc::shape() const { return body_desc_.shape(); }

DataType RtBlobDesc::data_type() const { return body_desc_.data_type(); }

size_t RtBlobDesc::ByteSizeOfBlobHeader() const { return header_pod_desc_.ByteSize(); }

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return body_desc_.AlignedByteSize(); }

size_t RtBlobDesc::ByteSizeOfDataIdField() const {
  if (!HasField<FieldKey::kDataId>()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kDataId).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfColNumField() const {
  if (!HasField<FieldKey::kColNum>()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kColNum).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfDim0ValidNumField() const {
  if (!HasField<FieldKey::kDim0ValidNum>()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kDim0ValidNum).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfDim1ValidNumField() const {
  if (!HasField<FieldKey::kDim1ValidNum>()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kDim1ValidNum).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfDim2ValidNumField() const {
  if (!HasField<FieldKey::kDim2ValidNum>()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kDim2ValidNum).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfRecordIdInDevicePieceField() const {
  if (!HasField<FieldKey::kRecordIdInDevicePiece>()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kRecordIdInDevicePiece).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfDataContentField() const { return body_desc_.ByteSize(); }

size_t RtBlobDesc::TotalByteSize() const { return ByteSizeOfBlobHeader() + ByteSizeOfBlobBody(); }

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  PbMd message_diff;
  return message_diff.Equals(blob_desc_proto_, rhs.blob_desc_proto());
}

}  // namespace oneflow
