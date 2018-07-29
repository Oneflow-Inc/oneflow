#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

CellDesc::CellDesc() : CellDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType()) {}

CellDesc::CellDesc(const Shape& shape, DataType data_type) : shape_(shape), data_type_(data_type) {}

CellDesc::CellDesc(const CellDescProto& proto) {
  shape_ = Shape(proto.shape());
  data_type_ = proto.data_type();
}

void CellDesc::ToProto(CellDescProto* proto) const {
  shape_.ToProto(proto->mutable_shape());
  proto->set_data_type(data_type_);
}

bool CellDesc::operator==(const CellDesc& rhs) const {
  return shape() == rhs.shape() && data_type() == rhs.data_type();
}

BlobHeaderDesc::BlobHeaderDesc(bool is_packed, bool has_data_id_field, bool has_col_num_field,
                               int32_t max_col_num, int64_t header_byte_size)
    : is_packed_(is_packed),
      has_data_id_field_(has_data_id_field),
      has_col_num_field_(has_col_num_field),
      max_col_num_(max_col_num),
      header_byte_size_(header_byte_size) {}

BlobHeaderDesc::BlobHeaderDesc(const BlobHeaderDescProto& proto) {
  is_packed_ = proto.is_packed();
  has_data_id_field_ = proto.has_data_id_field();
  has_col_num_field_ = proto.has_col_num_field();
  max_col_num_ = proto.max_col_num();
  header_byte_size_ = proto.header_byte_size();
}

void BlobHeaderDesc::ToProto(BlobHeaderDescProto* proto) const {
  proto->set_is_packed(is_packed_);
  proto->set_has_data_id_field(has_data_id_field_);
  proto->set_has_col_num_field(has_col_num_field_);
  proto->set_max_col_num(max_col_num_);
  if (header_byte_size_ >= 0) { proto->set_header_byte_size(header_byte_size_); }
}

bool BlobHeaderDesc::operator==(const BlobHeaderDesc& rhs) const {
  return is_packed_ == rhs.is_packed() && has_data_id_field_ == rhs.has_data_id_field()
         && has_col_num_field_ == rhs.has_col_num_field() && max_col_num_ == rhs.max_col_num()
         && header_byte_size_ == rhs.header_byte_size();
}

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType(), false, false, 1) {}

BlobDesc::BlobDesc(const Shape& shape, DataType data_type, bool has_data_id_field,
                   bool has_col_num_field, int32_t max_col_num)
    : body_desc_(shape, data_type),
      header_desc_(false, has_data_id_field, has_col_num_field, max_col_num, -1) {}

BlobDesc::BlobDesc(const BlobDescProto& proto)
    : header_desc_(proto.header_desc()), body_desc_(proto.body_desc()) {}

BlobDesc::BlobDesc(int64_t header_byte_size, int64_t body_byte_size, int32_t max_col_num)
    : header_desc_(true, false, false, max_col_num, header_byte_size),
      body_desc_(Shape({body_byte_size}), DataType::kChar) {
  CHECK_GT(header_byte_size, 0);
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  header_desc_.ToProto(proto->mutable_header_desc());
  body_desc_.ToProto(proto->mutable_body_desc());
}

size_t BlobDesc::ByteSizeOfBlobHeader() const {
  if (header_desc_.header_byte_size() > 0) {
    return header_desc_.header_byte_size();
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
  return header_desc_ == rhs.header_desc() && body_desc_ == rhs.body_desc();
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
    header_byte_size += blob_desc->ByteSizeOfBlobHeader();
    body_byte_size += blob_desc->ByteSizeOfBlobBody();
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
