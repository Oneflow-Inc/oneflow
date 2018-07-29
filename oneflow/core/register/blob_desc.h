#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class BlobHeaderDesc {
 public:
  ~BlobHeaderDesc() = default;

  void ToProto(BlobHeaderDescProto* proto) const;

 private:
};

class BlobBodyDesc {
 public:
  ~BlobBodyDesc() = default;

  BlobBodyDesc();
  BlobBodyDesc(const Shape& shape, DataType data_type);
  BlobBodyDesc(const Shape& shape) : BlobBodyDesc() { shape_ = shape; }
  BlobBodyDesc(const BlobBodyDescProto& proto);

  const Shape& shape() const { return shape_; }
  Shape& mut_shape() { return shape_; }

  DataType data_type() const { return data_type_; }
  void set_data_type(DataType val) { data_type_ = val; }

  void ToProto(BlobBodyDescProto* proto) const;

 private:
  Shape shape_;
  DataType data_type_;
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

  const Shape& shape() const { return body_desc_.shape(); }
  Shape& mut_shape() { return body_desc_.mut_shape(); }

  DataType data_type() const { return body_desc_.data_type(); }
  void set_data_type(DataType val) { body_desc_.set_data_type(val); }

  virtual bool has_blob_header() const { return has_data_id_field_ || has_col_num_field_; }

  bool has_data_id_field() const { return has_data_id_field_; }
  void set_has_data_id_field(bool val) { has_data_id_field_ = val; }

  bool has_col_num_field() const { return has_col_num_field_; }
  void set_has_col_num_field(bool val) { has_col_num_field_ = val; }

  int32_t max_col_num() const { return max_col_num_; }
  void set_max_col_num(int32_t val) { max_col_num_ = val; }

  virtual void ToProto(BlobDescProto* proto) const;
  virtual size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDataContentField() const;
  size_t TotalByteSize() const;
  virtual bool IsMemBlobDesc() const { return false; };
  bool operator==(const BlobDesc& rhs) const;

 private:
  BlobHeaderDesc header_desc_;
  BlobBodyDesc body_desc_;
  // Shape shape_;
  // DataType data_type_;
  bool has_data_id_field_;
  bool has_col_num_field_;
  int64_t max_col_num_;
};

class MemBlobDesc : public BlobDesc {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(MemBlobDesc);
  MemBlobDesc() = delete;
  ~MemBlobDesc() = default;

  MemBlobDesc(size_t header_byte_size, size_t body_byte_size, int32_t max_col_num);
  MemBlobDesc(const BlobDescProto& proto);

  void ToProto(BlobDescProto* proto) const override;
  bool has_blob_header() const override { return true; }
  size_t ByteSizeOfBlobHeader() const override { return header_byte_size_; }
  bool IsMemBlobDesc() const override { return true; };

 private:
  size_t header_byte_size_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(std::function<const BlobDesc*()> NextBlobDesc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
