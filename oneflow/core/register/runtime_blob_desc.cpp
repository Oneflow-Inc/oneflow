#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

RtBlobDesc::RtBlobDesc(const BlobDesc& blob_desc)
    : body_desc_(Shape({}), DataType::kInvalidDataType) {
  BlobDescProto blob_desc_proto;
  blob_desc.ToProto(&blob_desc_proto);
  InitFromProto(blob_desc_proto);
}

RtBlobDesc::RtBlobDesc(const BlobDescProto& blob_desc_proto)
    : body_desc_(Shape({}), DataType::kInvalidDataType) {
  InitFromProto(blob_desc_proto);
}

void RtBlobDesc::InitFromProto(const BlobDescProto& blob_desc_proto) {
  blob_desc_proto_ = blob_desc_proto;
  body_desc_ = FieldDesc(blob_desc_proto.body());
  header_pod_desc_.InitFromProto(blob_desc_proto.header().header_pod_desc());
  if (blob_desc_proto.has_dim0_inner_shape()) {
    dim0_inner_shape_.reset(new Shape(blob_desc_proto.dim0_inner_shape()));
  }
  has_dim0_valid_num_field_ = header_pod_desc_.HasField(FieldKey::kDim0ValidNum);
  if (has_dim0_valid_num_field_) {
    byte_size_of_dim0_valid_num_field_ = header_pod_desc_.Field(FieldKey::kDim0ValidNum).ByteSize();
  } else {
    byte_size_of_dim0_valid_num_field_ = 0;
  }
  byte_size_of_data_content_field_ = body_desc_.ByteSize();
}

const Shape& RtBlobDesc::shape() const { return body_desc_.shape(); }

DataType RtBlobDesc::data_type() const { return body_desc_.data_type(); }

bool RtBlobDesc::has_data_id_field() const { return header_pod_desc_.HasField(FieldKey::kDataId); }

bool RtBlobDesc::has_col_num_field() const { return header_pod_desc_.HasField(FieldKey::kColNum); }

bool RtBlobDesc::has_dim0_valid_num_field() const { return has_dim0_valid_num_field_; }

bool RtBlobDesc::has_dim1_valid_num_field() const {
  return header_pod_desc_.HasField(FieldKey::kDim1ValidNum);
}

bool RtBlobDesc::has_dim2_valid_num_field() const {
  return header_pod_desc_.HasField(FieldKey::kDim2ValidNum);
}

bool RtBlobDesc::has_record_id_in_device_piece_field() const {
  return header_pod_desc_.HasField(FieldKey::kRecordIdInDevicePiece);
}

bool RtBlobDesc::is_body_disabled() const { return blob_desc_proto_.is_body_disabled(); }

size_t RtBlobDesc::ByteSizeOfBlobHeader() const { return header_pod_desc_.ByteSize(); }

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return body_desc_.AlignedByteSize(); }

size_t RtBlobDesc::ByteSizeOfDataIdField() const {
  if (!has_data_id_field()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kDataId).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfColNumField() const {
  if (!has_col_num_field()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kColNum).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfDim0ValidNumField() const {
  return byte_size_of_dim0_valid_num_field_;
}

size_t RtBlobDesc::ByteSizeOfDim1ValidNumField() const {
  if (!has_dim1_valid_num_field()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kDim1ValidNum).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfDim2ValidNumField() const {
  if (!has_dim2_valid_num_field()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kDim2ValidNum).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfRecordIdInDevicePieceField() const {
  if (!has_record_id_in_device_piece_field()) { return 0; }
  return header_pod_desc_.Field(FieldKey::kRecordIdInDevicePiece).ByteSize();
}

size_t RtBlobDesc::ByteSizeOfDataContentField() const { return byte_size_of_data_content_field_; }

size_t RtBlobDesc::TotalByteSize() const { return ByteSizeOfBlobHeader() + ByteSizeOfBlobBody(); }

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  PbMd message_diff;
  return message_diff.Equals(blob_desc_proto_, rhs.blob_desc_proto());
}

}  // namespace oneflow
