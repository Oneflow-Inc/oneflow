#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobHeaderDesc::BlobHeaderDesc(bool header_is_opaque, bool has_data_id, bool has_col_num,
                               int32_t max_col_num)
    : header_is_opaque_(header_is_opaque),
      has_data_id_(has_data_id),
      has_col_num_(has_col_num),
      max_col_num_(max_col_num) {}

BlobHeaderDesc::BlobHeaderDesc(const BlobHeaderDescProto& proto) {
  header_is_opaque_ = proto.header_is_opaque();
  has_data_id_ = proto.has_data_id();
  has_col_num_ = proto.has_col_num();
  max_col_num_ = proto.max_col_num();
}

void BlobHeaderDesc::ToProto(BlobHeaderDescProto* proto) const {
  proto->set_header_is_opaque(header_is_opaque_);
  proto->set_has_data_id(has_data_id_);
  proto->set_has_col_num(has_col_num_);
  proto->set_max_col_num(max_col_num_);
}

bool BlobHeaderDesc::operator==(const BlobHeaderDesc& rhs) const {
  return header_is_opaque_ == rhs.header_is_opaque() && has_data_id_ == rhs.has_data_id()
         && has_col_num_ == rhs.has_col_num() && max_col_num_ == rhs.max_col_num();
}

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType(), false, false, 1) {}

BlobDesc::BlobDesc(const Shape& shape, DataType data_type, bool has_data_id, bool has_col_num,
                   int32_t max_col_num)
    : header_desc_(false, has_data_id, has_col_num, max_col_num), body_field_(shape, data_type) {}

BlobDesc::BlobDesc(const BlobDescProto& proto)
    : header_desc_(proto.header_desc()), body_field_(proto.body_field()) {}

BlobDesc::BlobDesc(int64_t header_byte_size, int64_t body_byte_size, int32_t max_col_num)
    : header_desc_(true, false, false, max_col_num),
      header_field_(Shape({header_byte_size}), DataType::kChar),
      body_field_(Shape({body_byte_size}), DataType::kChar) {
  CHECK_GE(header_byte_size, 0);
}

void BlobDesc::DataIdFieldToProto(BlobDescProto* proto) const {
  FieldDesc data_id_field(
      Shape({body_field_.shape().At(0), Global<JobDesc>::Get()->SizeOfOneDataId()}),
      DataType::kChar);
  data_id_field.ToProto(proto->mutable_data_id_field());
}

void BlobDesc::ColNumFieldToProto(BlobDescProto* proto) const {
  FieldDesc col_num_field(Shape({body_field_.shape().At(0)}), DataType::kInt32);
  col_num_field.ToProto(proto->mutable_col_num_field());
}

void BlobDesc::HeaderFieldToProto(BlobDescProto* proto) const {
  if (!header_desc_.header_is_opaque()) {
    FieldDesc header_field(Shape({ByteSizeOfBlobHeader()}), DataType::kChar);
    header_field.ToProto(proto->mutable_header_field());
  } else {
    header_field_.ToProto(proto->mutable_header_field());
  }
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  header_desc_.ToProto(proto->mutable_header_desc());
  body_field_.ToProto(proto->mutable_body_field());
  HeaderFieldToProto(proto);
  if (has_data_id_field()) { DataIdFieldToProto(proto); }
  if (has_col_num_field()) { ColNumFieldToProto(proto); }
}

size_t BlobDesc::ByteSizeOfBlobHeader() const {
  if (header_desc_.header_is_opaque()) {
    return header_field_.ByteSize();
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
  return header_desc_ == rhs.header_desc_ && body_field_ == rhs.body_field_;
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
