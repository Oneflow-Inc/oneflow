#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

class BlobDesc final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BlobDesc);
  BlobDesc();
  ~BlobDesc() = default;

  BlobDesc(const BlobDescProto& proto) {
    shape_ = Shape(proto.shape());
    data_type_ = proto.data_type();
    has_data_id_ = proto.has_data_id();
    max_seq_size_ = proto.max_seq_size();
  }
  BlobDesc(Shape shape, DataType data_type, bool has_data_id,
           int32_t max_seq_size)
      : shape_(shape),
        data_type_(data_type),
        has_data_id_(has_data_id),
        max_seq_size_(max_seq_size) {}

  const Shape& shape() const { return shape_; }
  Shape& mut_shape() { return shape_; }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType val) { data_type_ = val; }

  bool has_data_id() const { return has_data_id_; }
  void set_has_data_id(bool val) { has_data_id_ = val; }

  int32_t max_seq_size() const { return max_seq_size_; }
  void set_max_seq_size(int32_t val) { max_seq_size_ = val; }

  void ToProto(BlobDescProto* proto) const {
    shape_.ToProto(proto->mutable_shape());
    proto->set_data_type(data_type_);
    proto->set_has_data_id(has_data_id_);
    proto->set_max_seq_size(max_seq_size_);
  }
  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfDataContentField() const;
  size_t TotalByteSize() const;
  bool operator==(const BlobDesc& rhs) const;

 private:
  Shape shape_;
  DataType data_type_;
  bool has_data_id_;
  int32_t max_seq_size_;
};

BlobDesc ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
