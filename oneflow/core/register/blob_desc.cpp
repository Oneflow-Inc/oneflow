#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), JobDesc::Singleton()->DefaultDataType(), false, false,
               1) {}

BlobDesc::BlobDesc(Shape shape, DataType data_type, bool has_data_id_field,
                   bool has_col_num_field, int32_t max_col_num)
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

size_t BlobDesc::ByteSizeOfDataIdField() const {
  if (has_data_id_field_) {
    return shape_.At(0) * JobDesc::Singleton()->SizeOfOneDataId();
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

size_t BlobDesc::TotalByteSize() const {
  return ByteSizeOfDataIdField() + ByteSizeOfColNumField()
         + ByteSizeOfDataContentField();
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return shape_ == rhs.shape_ && data_type_ == rhs.data_type_
         && has_data_id_field_ == rhs.has_data_id_field_
         && has_col_num_field_ == rhs.has_col_num_field_
         && max_col_num_ == rhs.max_col_num_;
}

BlobDesc ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc) {
  int64_t total_byte_size = 0;
  int64_t total_data_content_byte_size = 0;
  HashSet<int> data_type_set;
  bool has_data_id_field = false;
  bool has_col_num_field = false;
  int32_t max_col_num = -1;
  int32_t blob_desc_cnt = 0;
  BlobDesc ret;
  while (const BlobDesc* blob_desc = NextBlobDesc()) {
    total_byte_size += blob_desc->TotalByteSize();
    total_data_content_byte_size += blob_desc->ByteSizeOfDataContentField();
    data_type_set.insert(static_cast<int>(blob_desc->data_type()));
    has_data_id_field = has_data_id_field || blob_desc->has_data_id_field();
    has_col_num_field = has_col_num_field || blob_desc->has_col_num_field();
    if (max_col_num == -1) {
      max_col_num = blob_desc->max_col_num();
    } else {
      CHECK_EQ(max_col_num, 1);
      CHECK_EQ(blob_desc->max_col_num(), 1);
    }
    blob_desc_cnt += 1;
    ret = *blob_desc;
  }
  if (blob_desc_cnt <= 1) { return ret; }
  CHECK_EQ(has_col_num_field, false);
  CHECK_EQ(max_col_num, 1);
  if (has_data_id_field == false && data_type_set.size() == 1) {
    DataType sole_data_type = static_cast<DataType>(*(data_type_set.begin()));
    int64_t size_of_one_elem = GetSizeOfDataType(sole_data_type);
    CHECK_EQ(total_data_content_byte_size % size_of_one_elem, 0);
    ret.mut_shape() = Shape({total_data_content_byte_size / size_of_one_elem});
    ret.set_data_type(sole_data_type);
  } else {
    ret.mut_shape() = Shape({total_byte_size});
    ret.set_data_type(DataType::kChar);
  }
  ret.set_has_data_id_field(false);
  ret.set_has_col_num_field(false);
  ret.set_max_col_num(1);
  return ret;
}

}  // namespace oneflow
