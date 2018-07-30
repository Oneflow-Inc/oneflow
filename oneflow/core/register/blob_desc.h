#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class BlobHeaderDesc {
 public:
  ~BlobHeaderDesc() = default;

  BlobHeaderDesc() : BlobHeaderDesc(false, false, false, 1) {}
  BlobHeaderDesc(bool heaer_is_packed, bool has_data_id, bool has_col_num, int32_t max_col_num);
  BlobHeaderDesc(const BlobHeaderDescProto& proto);

  bool header_is_packed() const { return header_is_packed_; }

  bool has_data_id() const { return has_data_id_; }
  void set_has_data_id(bool val) {
    CHECK(!header_is_packed_);
    has_data_id_ = val;
  }

  bool has_col_num() const { return has_col_num_; }
  void set_has_col_num(bool val) {
    CHECK(!header_is_packed_);
    has_col_num_ = val;
  }

  int32_t max_col_num() const { return max_col_num_; }
  void set_max_col_num(int32_t val) { max_col_num_ = val; }

  void ToProto(BlobHeaderDescProto* proto) const;
  bool operator==(const BlobHeaderDesc& rhs) const;

  std::string DebugStr() const {
    return std::to_string(has_data_id_) + "," + std::to_string(has_col_num_) + ","
           + std::to_string(max_col_num_);
  }

 private:
  bool header_is_packed_;
  bool has_data_id_;
  bool has_col_num_;
  int64_t max_col_num_;
};

class BlobDesc {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BlobDesc);
  ~BlobDesc() = default;

  BlobDesc();
  BlobDesc(const Shape&, DataType, bool has_data_id, bool has_col_num, int32_t max_col_num);
  BlobDesc(const Shape& shape) : body_field_(shape) {}
  BlobDesc(const BlobDescProto& proto);
  BlobDesc(int64_t header_byte_size, int64_t body_byte_size, int32_t max_col_num);

  const Shape& shape() const { return body_field_.shape(); }
  Shape& mut_shape() { return body_field_.mut_shape(); }

  DataType data_type() const { return body_field_.data_type(); }
  void set_data_type(DataType val) { body_field_.set_data_type(val); }

  bool has_data_id_field() const { return header_desc_.has_data_id(); }
  void set_has_data_id_field(bool val) { header_desc_.set_has_data_id(val); }

  bool has_col_num_field() const { return header_desc_.has_col_num(); }
  void set_has_col_num_field(bool val) { header_desc_.set_has_col_num(val); }

  int32_t max_col_num() const { return header_desc_.max_col_num(); }
  void set_max_col_num(int32_t val) { header_desc_.set_max_col_num(val); }

  void ToProto(BlobDescProto* proto) const;

  bool IsPackedHeader() const { return header_desc_.header_is_packed(); };

  bool operator==(const BlobDesc& rhs) const;
  std::string DebugStr() const {
    return header_desc_.DebugStr() + "," + body_field_.DebugStr() + ","
           + std::to_string(IsPackedHeader());
  }

 private:
  void DataIdFieldToProto(BlobDescProto* proto) const;
  void ColNumFieldToProto(BlobDescProto* proto) const;
  void HeaderFieldToProto(BlobDescProto* proto) const;

  size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDataContentField() const;
  size_t TotalByteSize() const;

  BlobHeaderDesc header_desc_;
  FieldDesc header_field_;
  FieldDesc body_field_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
