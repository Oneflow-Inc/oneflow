#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType(), false, false, 1) {}

BlobDesc::BlobDesc(const Shape& shape, DataType data_type, bool has_data_id, bool has_col_num,
                   int32_t max_col_num)
    : header_is_opaque_(false),
      has_data_id_(has_data_id),
      has_col_num_(has_col_num),
      max_col_num_(max_col_num),
      body_field_(shape, data_type) {}

BlobDesc::BlobDesc(const BlobDescProto& proto) : body_field_(proto.body()) {
  max_col_num_ = proto.header().max_col_num();
  if (proto.header().has_opaque_header()) {
    header_is_opaque_ = true;
    has_data_id_ = false;
    has_col_num_ = false;
    opaque_header_ = FieldDesc(proto.header().opaque_header());
  } else {
    CHECK(proto.header().has_field_header());
    header_is_opaque_ = false;
    has_data_id_ = proto.header().field_header().has_data_id();
    has_col_num_ = proto.header().field_header().has_col_num();
  }
}

BlobDesc::BlobDesc(int64_t header_byte_size, int64_t body_byte_size, int32_t max_col_num)
    : header_is_opaque_(true),
      opaque_header_(Shape({header_byte_size}), DataType::kChar),
      has_data_id_(false),
      has_col_num_(false),
      max_col_num_(max_col_num),
      body_field_(Shape({body_byte_size}), DataType::kChar) {
  CHECK_GE(header_byte_size, 0);
}

void BlobDesc::DataIdFieldToProto(FieldHeaderDesc* proto) const {
  FieldDesc data_id_field(
      Shape({body_field_.shape().At(0), Global<JobDesc>::Get()->SizeOfOneDataId()}),
      DataType::kChar);
  data_id_field.ToProto(proto->mutable_data_id());
}

void BlobDesc::ColNumFieldToProto(FieldHeaderDesc* proto) const {
  FieldDesc col_num_field(Shape({body_field_.shape().At(0)}), DataType::kInt32);
  col_num_field.ToProto(proto->mutable_col_num());
}

void BlobDesc::HeaderToProto(BlobDescProto* proto) const {
  proto->mutable_header()->set_max_col_num(max_col_num_);
  if (!header_is_opaque_) {
    FieldHeaderDesc* field_header = proto->mutable_header()->mutable_field_header();
    if (has_data_id_field()) { DataIdFieldToProto(field_header); }
    if (has_col_num_field()) { ColNumFieldToProto(field_header); }
  } else {
    opaque_header_.ToProto(proto->mutable_header()->mutable_opaque_header());
  }
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  HeaderToProto(proto);
  body_field_.ToProto(proto->mutable_body());
}

size_t BlobDesc::ByteSizeOfBlobHeader() const {
  if (header_is_opaque_) {
    return opaque_header_.ByteSize();
  } else {
    return ByteSizeOfDataIdField() + ByteSizeOfColNumField();
  }
}

size_t BlobDesc::ByteSizeOfBlobBody() const {
  return RoundUp(ByteSizeOfDataContentField(), kCudaAlignSize);
}

size_t BlobDesc::ByteSizeOfDataIdField() const {
  if (has_data_id_field()) {
    return shape().At(0) * Global<JobDesc>::Get()->SizeOfOneDataId();
  } else {
    return 0;
  }
}

size_t BlobDesc::ByteSizeOfColNumField() const {
  if (has_col_num_field()) {
    return shape().At(0) * sizeof(int32_t);
  } else {
    return 0;
  }
}

size_t BlobDesc::ByteSizeOfDataContentField() const {
  return shape().elem_cnt() * GetSizeOfDataType(data_type());
}

size_t BlobDesc::TotalByteSize() const { return ByteSizeOfBlobHeader() + ByteSizeOfBlobBody(); }

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return header_is_opaque_ == rhs.header_is_opaque_ && opaque_header_ == rhs.opaque_header_
         && has_data_id_ == rhs.has_data_id_ && has_col_num_ == rhs.has_col_num_
         && max_col_num_ == rhs.max_col_num_ && body_field_ == rhs.body_field_;
}

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc) {
  int64_t header_byte_size = 0;
  int64_t body_byte_size = 0;
  HashSet<int> data_type_set;
  int32_t max_col_num = -1;
  int32_t blob_desc_cnt = 0;
  std::unique_ptr<BlobDesc> ret(new BlobDesc());
  const BlobDesc* last_blob_desc = nullptr;
  while (const BlobDesc* blob_desc = NextBlobDesc()) {
    BlobDescProto blob_desc_proto;
    blob_desc->ToProto(&blob_desc_proto);
    RtBlobDesc rt_blob_desc(blob_desc_proto);
    header_byte_size += rt_blob_desc.ByteSizeOfBlobHeader();
    body_byte_size += rt_blob_desc.ByteSizeOfBlobBody();
    data_type_set.insert(static_cast<int>(blob_desc->data_type()));
    if (max_col_num == -1) {
      max_col_num = blob_desc->max_col_num();
    } else {
      CHECK_EQ(max_col_num, blob_desc->max_col_num());
    }
    blob_desc_cnt += 1;
    last_blob_desc = blob_desc;
  }
  if (blob_desc_cnt == 0) { return ret; }
  if (blob_desc_cnt == 1) {
    ret.reset(new BlobDesc(*last_blob_desc));
    return ret;
  }
  if (header_byte_size == 0 && data_type_set.size() == 1) {
    DataType sole_data_type = static_cast<DataType>(*(data_type_set.begin()));
    int64_t size_of_one_elem = GetSizeOfDataType(sole_data_type);
    CHECK_EQ(body_byte_size % size_of_one_elem, 0);
    ret.reset(new BlobDesc(Shape({body_byte_size / size_of_one_elem}), sole_data_type, false, false,
                           max_col_num));
    return ret;
  }
  ret.reset(new BlobDesc(header_byte_size, body_byte_size, max_col_num));
  return ret;
}

}  // namespace oneflow
