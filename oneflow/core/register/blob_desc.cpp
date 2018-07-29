#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobBodyDesc::BlobBodyDesc() : BlobBodyDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType()) {}

BlobBodyDesc::BlobBodyDesc(const Shape& shape, DataType data_type)
    : shape_(shape), data_type_(data_type) {}

BlobBodyDesc::BlobBodyDesc(const BlobBodyDescProto& proto) {
  shape_ = Shape(proto.shape());
  data_type_ = proto.data_type();
}

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType(), false, false, 1) {}

BlobDesc::BlobDesc(Shape shape, DataType data_type, bool has_data_id_field, bool has_col_num_field,
                   int32_t max_col_num)
    : shape_(shape),
      data_type_(data_type),
      has_data_id_field_(has_data_id_field),
      has_col_num_field_(has_col_num_field),
      max_col_num_(max_col_num) {}

BlobDesc::BlobDesc(const BlobDescProto& proto) {
  shape_ = Shape(proto.shape());
  data_type_ = proto.data_type();
  has_data_id_field_ = proto.has_data_id_field();
  has_col_num_field_ = proto.has_col_num_field();
  max_col_num_ = proto.max_col_num();
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  shape_.ToProto(proto->mutable_shape());
  proto->set_data_type(data_type_);
  proto->set_has_data_id_field(has_data_id_field_);
  proto->set_has_col_num_field(has_col_num_field_);
  proto->set_max_col_num(max_col_num_);
}

size_t BlobDesc::ByteSizeOfBlobHeader() const {
  return ByteSizeOfDataIdField() + ByteSizeOfColNumField();
}

size_t BlobDesc::ByteSizeOfBlobBody() const {
  return RoundUp(ByteSizeOfDataContentField(), kCudaAlignSize);
}

size_t BlobDesc::ByteSizeOfDataIdField() const {
  if (has_data_id_field_) {
    return shape_.At(0) * Global<JobDesc>::Get()->SizeOfOneDataId();
  } else {
    return 0;
  }
}

size_t BlobDesc::ByteSizeOfColNumField() const {
  if (has_col_num_field_) {
    return shape_.At(0) * sizeof(int32_t);
  } else {
    return 0;
  }
}

size_t BlobDesc::ByteSizeOfDataContentField() const {
  return shape_.elem_cnt() * GetSizeOfDataType(data_type_);
}

size_t BlobDesc::TotalByteSize() const { return ByteSizeOfBlobHeader() + ByteSizeOfBlobBody(); }

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return shape_ == rhs.shape_ && data_type_ == rhs.data_type_
         && has_data_id_field_ == rhs.has_data_id_field_
         && has_col_num_field_ == rhs.has_col_num_field_ && max_col_num_ == rhs.max_col_num_
         && IsMemBlobDesc() == rhs.IsMemBlobDesc();
}

MemBlobDesc::MemBlobDesc(size_t header_byte_size, size_t body_byte_size, int32_t max_col_num)
    : BlobDesc(Shape({body_byte_size}), DataType::kChar, false, false, max_col_num),
      header_byte_size_(header_byte_size) {
  CHECK_GT(header_byte_size_, 0);
}

MemBlobDesc::MemBlobDesc(const BlobDescProto& proto) : BlobDesc(proto) {
  header_byte_size_ = proto.header_byte_size_for_mem_blob();
  CHECK_GT(header_byte_size_, 0);
}

void MemBlobDesc::ToProto(BlobDescProto* proto) const {
  BlobDesc::ToProto(proto);
  proto->set_header_byte_size_for_mem_blob(header_byte_size_);
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
  ret.reset(new MemBlobDesc(header_byte_size, body_byte_size, max_col_num));
  return ret;
}

}  // namespace oneflow
