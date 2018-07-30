#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

class RtBlobDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtBlobDesc);
  RtBlobDesc() = delete;
  ~RtBlobDesc() = default;

  RtBlobDesc(const BlobDescProto& blob_desc_proto);

  const BlobDescProto& blob_desc_proto() const { return blob_desc_; }
  const Shape& shape() const;  // body shape
  DataType data_type() const;  // body data type
  const Shape& shape(const std::string& field_name) const;
  DataType data_type(const std::string& field_name) const;

  bool has_data_id_field() const;
  bool has_col_num_field() const;

  bool IsPackedHeader() const { return header_desc_.is_packed(); }
  int32_t max_col_num() const { return header_desc_.max_col_num(); }

  size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t TotalByteSize() const;

  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDataContentField() const;

  bool operator==(const RtBlobDesc& rhs) const;

 private:
  HashMap<std::string, FieldDesc>::const_iterator GetFieldIteratorOrFail(
      const std::string& field_name) const;
  bool HasField(const std::string& field_name) const;
  size_t ByteSizeOfField(const std::string& field_name) const;
  size_t AlignedByteSizeOfField(const std::string& field_name) const;

  BlobDescProto blob_desc_;
  // bool is_packed_header_;
  // int32_t max_col_num_;

  BlobHeaderDesc header_desc_;
  HashMap<std::string, FieldDesc> field_name2desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
