#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

RtBlobDesc::RtBlobDesc(const BlobDescProto& blob_desc_proto)
    : blob_desc_(blob_desc_proto), header_desc_(blob_desc_proto.header_desc()) {
  CHECK(field_name2desc_.emplace("body", FieldDesc(blob_desc_proto.body_field())).second);
  CHECK(field_name2desc_.emplace("header", FieldDesc(blob_desc_proto.header_field())).second);
  if (blob_desc_proto.has_data_id_field()) {
    CHECK(field_name2desc_.emplace("data_id", FieldDesc(blob_desc_proto.data_id_field())).second);
  }
  if (blob_desc_proto.has_col_num_field()) {
    CHECK(field_name2desc_.emplace("col_num", FieldDesc(blob_desc_proto.col_num_field())).second);
  }
}

const Shape& RtBlobDesc::shape() const { return shape("body"); }

DataType RtBlobDesc::data_type() const { return data_type("body"); }

const Shape& RtBlobDesc::shape(const std::string& field_name) const {
  auto field_it = GetFieldIteratorOrFail(field_name);
  return field_it->second.shape();
}

DataType RtBlobDesc::data_type(const std::string& field_name) const {
  auto field_it = GetFieldIteratorOrFail(field_name);
  return field_it->second.data_type();
}

bool RtBlobDesc::has_data_id_field() const { return HasField("dara_id"); }

bool RtBlobDesc::has_col_num_field() const { return HasField("col_num"); }

size_t RtBlobDesc::ByteSizeOfBlobHeader() const { return ByteSizeOfField("header"); }

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return AlignedByteSizeOfField("body"); }

size_t RtBlobDesc::ByteSizeOfDataIdField() const {
  return HasField("data_id") ? ByteSizeOfField("data_id") : 0;
}

size_t RtBlobDesc::ByteSizeOfColNumField() const {
  return HasField("col_num") ? ByteSizeOfField("col_num") : 0;
}

size_t RtBlobDesc::ByteSizeOfDataContentField() const { return ByteSizeOfField("body"); }

size_t RtBlobDesc::TotalByteSize() const { return ByteSizeOfBlobHeader() + ByteSizeOfBlobBody(); }

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  PbMd message_diff;
  return message_diff.Equals(blob_desc_, rhs.blob_desc_proto());
}

HashMap<std::string, FieldDesc>::const_iterator RtBlobDesc::GetFieldIteratorOrFail(
    const std::string& field_name) const {
  auto field_it = field_name2desc_.find(field_name);
  CHECK(field_it != field_name2desc_.end());
  return field_it;
}

bool RtBlobDesc::HasField(const std::string& field_name) const {
  return field_name2desc_.find(field_name) != field_name2desc_.end();
}

size_t RtBlobDesc::ByteSizeOfField(const std::string& field_name) const {
  auto field_it = GetFieldIteratorOrFail(field_name);
  return field_it->second.ByteSize();
}

size_t RtBlobDesc::AlignedByteSizeOfField(const std::string& field_name) const {
  auto field_it = GetFieldIteratorOrFail(field_name);
  return field_it->second.AlignedByteSize();
}

}  // namespace oneflow
