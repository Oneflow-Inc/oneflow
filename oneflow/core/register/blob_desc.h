#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/register/pod_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class BlobDesc {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BlobDesc);
  ~BlobDesc() = default;

  BlobDesc();
  BlobDesc(const Shape&, DataType, bool has_data_id, bool has_col_num, int32_t max_col_num);
  explicit BlobDesc(const BlobDescProto& proto) { InitFromProto(proto); }
  explicit BlobDesc(const BlobDesc& blob_desc);
  BlobDesc(const StructPodDesc& header_pod_desc, int64_t header_byte_size, const Shape&, DataType,
           int32_t max_col_num);

  const Shape& shape() const { return body_field_.shape(); }
  Shape& mut_shape() { return body_field_.mut_shape(); }

  bool has_dim0_inner_shape() const { return bool(dim0_inner_shape_); }
  const Shape& dim0_inner_shape() const { return *dim0_inner_shape_; }
  Shape& mut_dim0_inner_shape();
  void clear_dim0_inner_shape() { dim0_inner_shape_.reset(nullptr); }

  DataType data_type() const { return body_field_.data_type(); }
  void set_data_type(DataType val) { body_field_.set_data_type(val); }

  bool header_is_opaque() const { return header_is_opaque_; };

  template<FieldKey field_key>
  bool HasField() const;

  template<FieldKey field_key>
  void SetHasField(bool val);

  bool has_data_id_field() const { return has_data_id_; }
  void set_has_data_id_field(bool val);

  bool has_dim0_valid_num_field() const { return has_dim0_valid_num_; }
  void set_has_dim0_valid_num_field(bool val);

  bool has_dim1_valid_num_field() const { return has_dim1_valid_num_; }
  void set_has_dim1_valid_num_field(bool val);

  bool has_dim2_valid_num_field() const { return has_dim2_valid_num_; }
  void set_has_dim2_valid_num_field(bool val);

  bool has_record_id_in_device_piece_field() const { return has_record_id_in_device_piece_; }
  void set_has_record_id_in_device_piece_field(bool val);

  bool has_col_num_field() const { return has_col_num_; }
  void set_has_col_num_field(bool val);

  int32_t max_col_num() const { return max_col_num_; }
  void set_max_col_num(int32_t val) { max_col_num_ = val; }

  int32_t blob_mem_id() const { return blob_mem_id_; }
  void set_blob_mem_id(int32_t val) { blob_mem_id_ = val; }

  bool operator==(const BlobDesc& rhs) const;
  void ToProto(BlobDescProto* proto) const;
  BlobDesc& operator=(const BlobDesc& blob_desc);

 private:
  void InitFromProto(const BlobDescProto& proto);
  void HeaderToProto(BlobDescProto* proto) const;
  void DataIdFieldToProto(FieldHeaderDesc* proto, StructPodDesc* header_pod_desc) const;
  void ColNumFieldToProto(FieldHeaderDesc* proto, StructPodDesc* header_pod_desc) const;
  void Dim0ValidNumToProto(StructPodDesc* header_pod_desc) const;
  void Dim1ValidNumToProto(StructPodDesc* header_pod_desc) const;
  void Dim2ValidNumToProto(StructPodDesc* header_pod_desc) const;
  void RecordIdInDevicePieceToProto(StructPodDesc* header_pod_desc) const;

  bool header_is_opaque_;
  FieldDesc opaque_header_;
  StructPodDesc header_pod_desc_;

  bool has_data_id_;
  bool has_col_num_;
  bool has_dim0_valid_num_;
  bool has_dim1_valid_num_;
  bool has_dim2_valid_num_;
  bool has_record_id_in_device_piece_;
  int64_t max_col_num_;
  int32_t blob_mem_id_;

  FieldDesc body_field_;
  std::unique_ptr<Shape> dim0_inner_shape_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
