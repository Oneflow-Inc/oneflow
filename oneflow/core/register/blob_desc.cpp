#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), JobDesc::Singleton()->DefaultDataType(), false, false,
               1) {}

BlobDesc::BlobDesc(Shape shape, DataType data_type, bool has_data_id,
                   bool has_seq_len, int32_t max_seq_len)
    : shape_(shape),
      data_type_(data_type),
      has_data_id_(has_data_id),
      has_seq_len_(has_seq_len),
      max_seq_len_(max_seq_len) {}

BlobDesc::BlobDesc(const BlobDescProto& proto) {
  shape_ = Shape(proto.shape());
  data_type_ = proto.data_type();
  has_data_id_ = proto.has_data_id();
  has_seq_len_ = proto.has_seq_len();
  max_seq_len_ = proto.max_seq_len();
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  shape_.ToProto(proto->mutable_shape());
  proto->set_data_type(data_type_);
  proto->set_has_data_id(has_data_id_);
  proto->set_has_seq_len(has_seq_len_);
  proto->set_max_seq_len(max_seq_len_);
}

size_t BlobDesc::ByteSizeOfBlobHeaderField() const {
  return sizeof(BlobHeader);
}

size_t BlobDesc::ByteSizeOfDataIdField() const {
  if (has_data_id_) {
    return shape_.At(0) * JobDesc::Singleton()->SizeOfOneDataId();
  } else {
    return 0;
  }
}

size_t BlobDesc::ByteSizeOfSeqLenField() const {
  if (has_seq_len_) {
    return shape_.At(0) * sizeof(int32_t);
  } else {
    return 0;
  }
}

size_t BlobDesc::ByteSizeOfDataContentField() const {
  return shape_.elem_cnt() * GetSizeOfDataType(data_type_);
}

size_t BlobDesc::TotalByteSize() const {
  return ByteSizeOfBlobHeaderField() + ByteSizeOfDataIdField()
         + ByteSizeOfSeqLenField() + ByteSizeOfDataContentField();
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return shape_ == rhs.shape_ && data_type_ == rhs.data_type_
         && has_data_id_ == rhs.has_data_id_ && has_seq_len_ == rhs.has_seq_len_
         && max_seq_len_ == rhs.max_seq_len_;
}

BlobDesc ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc) {
  int64_t total_byte_size = 0;
  HashSet<int> data_type_set;
  bool has_data_id = false;
  int32_t seq_len_flag = 0;
  int32_t max_seq_len = -1;
  while (const BlobDesc* blob_desc = NextBlobDesc()) {
    total_byte_size += blob_desc->TotalByteSize();
    data_type_set.insert(static_cast<int>(blob_desc->data_type()));
    has_data_id = has_data_id || blob_desc->has_data_id();
    if (seq_len_flag == 0) {
      seq_len_flag = blob_desc->has_seq_len() ? 1 : -1;
    } else if (seq_len_flag == 1) {
      CHECK(blob_desc->has_seq_len());
    } else {
      CHECK(!(blob_desc->has_seq_len()));
    }
    if (max_seq_len == -1) {
      max_seq_len = blob_desc->max_seq_len();
    } else {
      CHECK_EQ(max_seq_len, 1);
      CHECK_EQ(blob_desc->max_seq_len(), 1);
    }
  }
  BlobDesc ret;
  if (has_data_id == false && seq_len_flag == -1 && data_type_set.size() == 1) {
    DataType sole_data_type = static_cast<DataType>(*(data_type_set.begin()));
    int64_t size_of_one_elem = GetSizeOfDataType(sole_data_type);
    CHECK_EQ(total_byte_size % size_of_one_elem, 0);
    ret.mut_shape() = Shape({total_byte_size / size_of_one_elem});
    ret.set_data_type(sole_data_type);
  } else {
    ret.mut_shape() = Shape({total_byte_size});
    ret.set_data_type(DataType::kChar);
  }
  ret.set_has_data_id(false);
  ret.set_has_seq_len(false);
  ret.set_max_seq_len(max_seq_len);
  return ret;
}

}  // namespace oneflow
