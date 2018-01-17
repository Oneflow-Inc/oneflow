#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

struct BlobHeader {
  int32_t col_id;
  int32_t max_col_id;
};

class BlobDesc final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BlobDesc);
  ~BlobDesc() = default;

  BlobDesc();
  BlobDesc(Shape, DataType, bool has_data_id_field, bool has_col_num_field,
           int32_t max_col_num);
  BlobDesc(Shape shape) : BlobDesc() { shape_ = shape; }
  BlobDesc(const BlobDescProto& proto);

  const Shape& shape() const { return shape_; }
  Shape& mut_shape() { return shape_; }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType val) { data_type_ = val; }

  bool has_data_id_field() const { return has_data_id_field_; }
  void set_has_data_id_field(bool val) { has_data_id_field_ = val; }

  bool has_col_num_field() const { return has_col_num_field_; }
  void set_has_col_num_field(bool val) { has_col_num_field_ = val; }

  int32_t max_col_num() const { return max_col_num_; }
  void set_max_col_num(int32_t val) { max_col_num_ = val; }

  void ToProto(BlobDescProto* proto) const;
  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDataContentField() const;
  size_t TotalByteSize() const;
  bool operator==(const BlobDesc& rhs) const;

 private:
  Shape shape_;
  DataType data_type_;
  bool has_data_id_field_;
  bool has_col_num_field_;
  int64_t max_col_num_;
};

BlobDesc ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
