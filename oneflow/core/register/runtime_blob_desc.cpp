#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

RtBlobDesc::RtBlobDesc(const BlobDescProto& blob_desc_proto) {}

const Shape& RtBlobDesc::shape() const { return shape("body"); }

DataType RtBlobDesc::data_type() const { return data_type("body"); }

const Shape& RtBlobDesc::shape(const std::string& field_name) const {
  auto field_it = field_name2desc_.find(field_name);
  CHECK(field_it != field_name2desc_.end());
  return field_it->second.shape();
}

DataType RtBlobDesc::data_type(const std::string& field_name) const {
  auto field_it = field_name2desc_.find(field_name);
  CHECK(field_it != field_name2desc_.end());
  return field_it->second.data_type();
}

bool RtBlobDesc::has_data_id() const { return HasField("dara_id"); }

bool RtBlobDesc::has_col_num() const { return HasField("col_num"); }

bool RtBlobDesc::has_blob_header() const {
  // TODO
  return true;
}

bool RtBlobDesc::IsPackedHeader() const {
  // TODO
  return true;
}

size_t RtBlobDesc::ByteSizeOfBlobHeader() const {
  // TODO
  return 0;
}

size_t RtBlobDesc::ByteSizeOfBlobBody() const { return AlignedByteSizeOfField("body"); }

size_t RtBlobDesc::ByteSizeOfDataId() const {
  if (HasField("data_id")) {
    return ByteSizeOfField("data_id");
  } else {
    return 0;
  }
}

size_t RtBlobDesc::ByteSizeOfColNum() const {
  if (HasField("col_num")) {
    return ByteSizeOfField("col_num");
  } else {
    return 0;
  }
}

size_t RtBlobDesc::ByteSizeOfBodyContent() const { return ByteSizeOfField("body"); }

size_t RtBlobDesc::TotalByteSize() const { return ByteSizeOfBlobHeader() + ByteSizeOfBlobBody(); }

bool RtBlobDesc::HasField(const std::string& field_name) const {
  return field_name2desc_.find(field_name) != field_name2desc_.end();
}

size_t RtBlobDesc::ByteSizeOfField(const std::string& field_name) const {
  auto field_it = field_name2desc_.find(field_name);
  CHECK(field_it != field_name2desc_.end());
  return field_it->second.ByteSize();
}

size_t RtBlobDesc::AlignedByteSizeOfField(const std::string& field_name) const {
  auto field_it = field_name2desc_.find(field_name);
  CHECK(field_it != field_name2desc_.end());
  return field_it->second.AlignedByteSize();
}

}  // namespace oneflow
