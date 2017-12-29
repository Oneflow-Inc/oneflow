#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobDesc::BlobDesc()
    : shape_(),
      data_type_(JobDesc::Singleton()->DefaultDataType()),
      has_data_id_(false),
      max_seq_size_(1) {}

size_t BlobDesc::ByteSizeOfDataIdField() const {
  if (has_data_id_) {
    return shape_.At(0) * JobDesc::Singleton()->SizeOfOneDataId();
  } else {
    return 0;
  }
}

size_t BlobDesc::ByteSizeOfDataContentField() const {
  return shape_.elem_cnt() * GetSizeOfDataType(data_type_);
}

size_t BlobDesc::TotalByteSize() const {
  return ByteSizeOfDataIdField() + ByteSizeOfDataContentField();
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return shape_ == rhs.shape_ && data_type_ == rhs.data_type_
         && has_data_id_ == rhs.has_data_id_
         && max_seq_size_ == rhs.max_seq_size_;
}

BlobDesc ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc) {
  int64_t total_byte_size = 0;
  std::unordered_set<int> data_type_set;
  bool has_data_id = false;
  int32_t max_seq_size = -1;
  while (const BlobDesc* blob_desc = NextBlobDesc()) {
    total_byte_size += blob_desc->TotalByteSize();
    data_type_set.insert(static_cast<int>(blob_desc->data_type()));
    has_data_id = has_data_id || blob_desc->has_data_id();
    if (max_seq_size == -1) {
      max_seq_size = blob_desc->max_seq_size();
    } else {
      CHECK_EQ(max_seq_size, 1);
      CHECK_EQ(blob_desc->max_seq_size(), 1);
    }
  }
  BlobDesc ret;
  if (has_data_id == false && data_type_set.size() == 1) {
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
  ret.set_max_seq_size(max_seq_size);
  return ret;
}

}  // namespace oneflow
