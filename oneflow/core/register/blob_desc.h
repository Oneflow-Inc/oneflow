#ifndef ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_BLOB_DESC_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/register/pod_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

#define FIELD_KEY_AND_FIELD_NAME_SEQ                            \
  OF_PP_MAKE_TUPLE_SEQ(FieldKey::kDataId, data_id)              \
  OF_PP_MAKE_TUPLE_SEQ(FieldKey::kColNum, col_num)              \
  OF_PP_MAKE_TUPLE_SEQ(FieldKey::kDim0ValidNum, dim0_valid_num) \
  OF_PP_MAKE_TUPLE_SEQ(FieldKey::kDim1ValidNum, dim1_valid_num) \
  OF_PP_MAKE_TUPLE_SEQ(FieldKey::kDim2ValidNum, dim2_valid_num) \
  OF_PP_MAKE_TUPLE_SEQ(FieldKey::kRecordIdInDevicePiece, record_id_in_device_piece)

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
  template<FieldKey field_key>
  void FieldToProto(FieldHeaderDesc* proto, StructPodDesc* header_pod_desc) const;

  bool header_is_opaque_;
  FieldDesc opaque_header_;
  StructPodDesc header_pod_desc_;

#define DEFINE_HAS_FIELD_MEMBER(field_key, field_name) bool has_##field_name##_;

  OF_PP_FOR_EACH_TUPLE(DEFINE_HAS_FIELD_MEMBER, FIELD_KEY_AND_FIELD_NAME_SEQ)

#undef DEFINE_HAS_FIELD_MEMBER

  int64_t max_col_num_;
  int32_t blob_mem_id_;

  FieldDesc body_field_;
  std::unique_ptr<Shape> dim0_inner_shape_;
};

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_DESC_H_
