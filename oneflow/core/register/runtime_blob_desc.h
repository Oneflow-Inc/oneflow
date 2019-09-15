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

  RtBlobDesc(const BlobDesc& blob_desc);
  RtBlobDesc(const BlobDescProto& blob_desc_proto);

  const BlobDescProto& blob_desc_proto() const { return blob_desc_proto_; }
  const Shape& shape() const;  // body shape
  DataType data_type() const;  // body data type
  bool has_dim0_inner_shape() const { return bool(dim0_inner_shape_); }
  const Shape& dim0_inner_shape() const { return *dim0_inner_shape_; }

  bool has_data_id_field() const;
  bool has_col_num_field() const;
  bool has_dim0_valid_num_field() const;
  bool has_dim1_valid_num_field() const;
  bool has_dim2_valid_num_field() const;
  bool has_record_id_in_device_piece_field() const;
  bool is_body_disabled() const;
  const StructPodDesc& header_pod_desc() const { return header_pod_desc_; }

  int32_t max_col_num() const { return blob_desc_proto_.header().max_col_num(); }

  size_t ByteSizeOfBlobHeader() const;
  size_t ByteSizeOfBlobBody() const;
  size_t TotalByteSize() const;

  size_t ByteSizeOfDataIdField() const;
  size_t ByteSizeOfColNumField() const;
  size_t ByteSizeOfDim0ValidNumField() const;
  size_t ByteSizeOfDim1ValidNumField() const;
  size_t ByteSizeOfDim2ValidNumField() const;
  size_t ByteSizeOfRecordIdInDevicePieceField() const;
  size_t ByteSizeOfDataContentField() const;

  bool operator==(const RtBlobDesc& rhs) const;

 private:
  void InitFromProto(const BlobDescProto& proto);

  BlobDescProto blob_desc_proto_;
  FieldDesc body_desc_;
  StructPodDesc header_pod_desc_;
  std::unique_ptr<Shape> dim0_inner_shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
