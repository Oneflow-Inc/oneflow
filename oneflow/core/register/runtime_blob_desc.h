#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

class RtBlobDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtBlobDesc);
  RtBlobDesc() = delete;
  ~RtBlobDesc() = default;

  RtBlobDesc(const BlobDescProto& blob_desc_proto);

  const Shape& shape() const;  // body shape
  DataType data_type() const;  // body data type
  const Shape& shape(const std::string& field_name) const;
  DataType data_type(const std::string& field_name) const;

  bool has_data_id() const;
  bool has_col_num() const;

  bool has_blob_header() const;
  bool IsPackedHeader() const;
  int32_t max_col_num() const { return max_col_num_; }

  size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t TotalByteSize() const;

  size_t ByteSizeOfDataId() const;
  size_t ByteSizeOfColNum() const;
  size_t ByteSizeOfBodyContent() const;

 private:
  HashMap<std::string, FieldDesc>::const_iterator GetFieldIteratorOrFail(
      const std::string& field_name) const;
  bool HasField(const std::string& field_name) const;
  size_t ByteSizeOfField(const std::string& field_name) const;
  size_t AlignedByteSizeOfField(const std::string& field_name) const;

  BlobDescProto blob_desc_;
  bool is_packed_header_;
  int32_t max_col_num_;
  HashMap<std::string, FieldDesc> field_name2desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
