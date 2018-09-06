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
  if (blob_desc_proto.header().has_opaque_header()) {
    CHECK(header_desc_.emplace("opaque_header", FieldDesc(blob_desc_proto.header().opaque_header()))
              .second);
  } else {
    CHECK(blob_desc_proto.header().has_field_header());
    if (blob_desc_proto.header().field_header().has_data_id()) {
      CHECK(header_desc_
                .emplace("data_id", FieldDesc(blob_desc_proto.header().field_header().data_id()))
                .second);
    }
    if (blob_desc_proto.header().field_header().has_col_num()) {
      CHECK(header_desc_
                .emplace("col_num", FieldDesc(blob_desc_proto.header().field_header().col_num()))
                .second);
    }
  }
}

const Shape& RtBlobDesc::shape() const { return body_desc_.shape(); }

DataType RtBlobDesc::data_type() const { return body_desc_.data_type(); }

const Shape& RtBlobDesc::shape(const std::string& field_name) const {
  auto field_it = GetFieldIteratorOrFail(field_name);
  return field_it->second.shape();
}

DataType RtBlobDesc::data_type(const std::string& field_name) const {
  auto field_it = GetFieldIteratorOrFail(field_name);
  return field_it->second.data_type();
}

bool RtBlobDesc::has_data_id_field() const { return HasField("data_id"); }

bool RtBlobDesc::has_col_num_field() const { return HasField("col_num"); }

bool RtBlobDesc::has_instance_num_field() const { return HasField("instance_num"); }

size_t RtBlobDesc::ByteSizeOfBlobHeader() const {
  size_t header_size = 0;
  for (auto& pair : header_desc_) { header_size += ByteSizeOfField(pair.first); }
  return header_size;
}

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return body_desc_.AlignedByteSize(); }

size_t RtBlobDesc::ByteSizeOfDataIdField() const {
  return HasField("data_id") ? ByteSizeOfField("data_id") : 0;
}

size_t RtBlobDesc::ByteSizeOfColNumField() const {
  return HasField("col_num") ? ByteSizeOfField("col_num") : 0;
}

size_t RtBlobDesc::ByteSizeOfInstanceNumField() const {
  return HasField("instance_num") ? ByteSizeOfField("instance_num") : 0;
}

size_t RtBlobDesc::ByteSizeOfDataContentField() const { return body_desc_.ByteSize(); }

size_t RtBlobDesc::TotalByteSize() const { return ByteSizeOfBlobHeader() + ByteSizeOfBlobBody(); }

bool RtBlobDesc::operator==(const RtBlobDesc& rhs) const {
  PbMd message_diff;
  return message_diff.Equals(blob_desc_proto_, rhs.blob_desc_proto());
}

HashMap<std::string, FieldDesc>::const_iterator RtBlobDesc::GetFieldIteratorOrFail(
    const std::string& field_name) const {
  auto field_it = header_desc_.find(field_name);
  CHECK(field_it != header_desc_.end());
  return field_it;
}

bool RtBlobDesc::HasField(const std::string& field_name) const {
  return header_desc_.find(field_name) != header_desc_.end();
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
