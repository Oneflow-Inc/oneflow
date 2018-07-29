#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class CellDesc {
 public:
  ~CellDesc() = default;

  CellDesc();
  CellDesc(const Shape& shape, DataType data_type);
  CellDesc(const Shape& shape) : CellDesc() { shape_ = shape; }
  CellDesc(const CellDescProto& proto);

  const Shape& shape() const { return shape_; }
  Shape& mut_shape() { return shape_; }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType val) { data_type_ = val; }

  void ToProto(CellDescProto* proto) const;

  std::string DebugStr() const { return shape_.DebugStr() + "," + std::to_string(data_type_); }

 private:
  Shape shape_;
  DataType data_type_;
};

using BlobBodyDesc = CellDesc;

class BlobHeaderDesc {
 public:
  ~BlobHeaderDesc() = default;

  BlobHeaderDesc() : BlobHeaderDesc(false, false, 1, -1) {}
  BlobHeaderDesc(bool has_data_id_field, bool has_col_num_filed, int32_t max_col_num,
                 int64_t header_byte_size);
  BlobHeaderDesc(const BlobHeaderDescProto& proto);

  bool has_data_id_field() const { return has_data_id_field_; }
  void set_has_data_id_field(bool val) { has_data_id_field_ = val; }

  bool has_col_num_field() const { return has_col_num_field_; }
  void set_has_col_num_field(bool val) { has_col_num_field_ = val; }

  int32_t max_col_num() const { return max_col_num_; }
  void set_max_col_num(int32_t val) { max_col_num_ = val; }

  int64_t header_byte_size() const { return header_byte_size_; }

  void ToProto(BlobHeaderDescProto* proto) const;

  std::string DebugStr() const {
    return std::to_string(has_data_id_field_) + "," + std::to_string(has_col_num_field_) + ","
           + std::to_string(max_col_num_);
  }

 private:
  bool has_data_id_field_;
  bool has_col_num_field_;
  int64_t max_col_num_;
  int64_t header_byte_size_;
};

class BlobDesc {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BlobDesc);
  ~BlobDesc() = default;

  BlobDesc();
  BlobDesc(const Shape&, DataType, bool has_data_id_field, bool has_col_num_field,
           int32_t max_col_num);
  BlobDesc(const Shape& shape) : body_desc_(shape) {}
  BlobDesc(const BlobDescProto& proto);
  BlobDesc(int64_t header_byte_size, int64_t body_byte_size, int32_t max_col_num);

  const Shape& shape() const { return body_desc_.shape(); }
  Shape& mut_shape() { return body_desc_.mut_shape(); }

  DataType data_type() const { return body_desc_.data_type(); }
  void set_data_type(DataType val) { body_desc_.set_data_type(val); }

  bool has_data_id_field() const { return header_desc_.has_data_id_field(); }
  void set_has_data_id_field(bool val) { header_desc_.set_has_data_id_field(val); }

  bool has_col_num_field() const { return header_desc_.has_col_num_field(); }
  void set_has_col_num_field(bool val) { header_desc_.set_has_col_num_field(val); }

  int32_t max_col_num() const { return header_desc_.max_col_num(); }
  void set_max_col_num(int32_t val) { header_desc_.set_max_col_num(val); }

  bool has_blob_header() const {
    return has_data_id_field() || has_col_num_field() || header_desc_.header_byte_size() > 0;
  }

  void ToProto(BlobDescProto* proto) const;
  size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDataContentField() const;
  size_t TotalByteSize() const;
  bool IsMemBlobDesc() const { return header_desc_.header_byte_size() > 0; };
  bool operator==(const BlobDesc& rhs) const;
  std::string DebugStr() const {
    return header_desc_.DebugStr() + "," + body_desc_.DebugStr() + ","
           + std::to_string(IsMemBlobDesc());
  }

 private:
  BlobHeaderDesc header_desc_;
  BlobBodyDesc body_desc_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
