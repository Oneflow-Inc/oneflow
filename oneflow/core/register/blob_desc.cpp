#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobDesc::BlobDesc(DataType data_type) : BlobDesc(Shape(), data_type, false, false, 1) {}

BlobDesc::BlobDesc(const Shape& shape, DataType data_type, bool has_data_id, bool has_col_num,
                   int32_t max_col_num)
    : header_is_opaque_(false),
      opaque_header_(Shape({}), data_type),
      has_data_id_(has_data_id),
      has_col_num_(has_col_num),
      has_dim0_valid_num_(false),
      has_dim1_valid_num_(false),
      has_dim2_valid_num_(false),
      has_record_id_in_device_piece_(false),
      max_col_num_(max_col_num),
      body_field_(shape, data_type),
      is_body_disabled_(false) {}

BlobDesc::BlobDesc(const BlobDesc& blob_desc) : BlobDesc(DataType::kInvalidDataType) {
  this->CopyAllFrom(blob_desc);
}

void BlobDesc::InitFromProto(const BlobDescProto& proto) {
  body_field_.InitFromProto(proto.body());
  max_col_num_ = proto.header().max_col_num();
  header_pod_desc_.InitFromProto(proto.header().header_pod_desc());
  is_body_disabled_ = proto.is_body_disabled();
  if (proto.header().has_opaque_header()) {
    header_is_opaque_ = true;
    has_data_id_ = false;
    has_col_num_ = false;
    has_dim0_valid_num_ = false;
    has_dim1_valid_num_ = false;
    has_dim2_valid_num_ = false;
    has_record_id_in_device_piece_ = false;
    opaque_header_ = FieldDesc(proto.header().opaque_header());
  } else {
    CHECK(proto.header().has_field_header());
    header_is_opaque_ = false;
    has_data_id_ = header_pod_desc_.HasField(FieldKey::kDataId);
    has_col_num_ = header_pod_desc_.HasField(FieldKey::kColNum);
    has_dim0_valid_num_ = header_pod_desc_.HasField(FieldKey::kDim0ValidNum);
    has_dim1_valid_num_ = header_pod_desc_.HasField(FieldKey::kDim1ValidNum);
    has_dim2_valid_num_ = header_pod_desc_.HasField(FieldKey::kDim2ValidNum);
    has_record_id_in_device_piece_ = header_pod_desc_.HasField(FieldKey::kRecordIdInDevicePiece);
  }
  if (proto.has_dim0_inner_shape()) {
    dim0_inner_shape_.reset(new Shape(proto.dim0_inner_shape()));
  }
}

BlobDesc::BlobDesc(const StructPodDesc& header_pod_desc, int64_t header_byte_size,
                   const Shape& shape, DataType data_type, int32_t max_col_num)
    : opaque_header_(Shape({}), DataType::kInvalidDataType),
      has_data_id_(false),
      has_col_num_(false),
      has_dim0_valid_num_(false),
      has_dim1_valid_num_(false),
      has_dim2_valid_num_(false),
      has_record_id_in_device_piece_(false),
      max_col_num_(max_col_num),
      body_field_(shape, data_type),
      is_body_disabled_(false) {
  CHECK_EQ(header_pod_desc.ByteSize(), header_byte_size);
  if (header_byte_size > 0) {
    header_is_opaque_ = true;
    opaque_header_ = FieldDesc(Shape({header_byte_size}), DataType::kChar);
    header_pod_desc_ = header_pod_desc;
  } else {
    header_is_opaque_ = false;
  }
}

void BlobDesc::set_has_data_id_field(bool val) {
  CHECK(!header_is_opaque_);
  has_data_id_ = val;
}

void BlobDesc::set_has_col_num_field(bool val) {
  CHECK(!header_is_opaque_);
  has_col_num_ = val;
}

void BlobDesc::set_has_dim0_valid_num_field(bool val) {
  CHECK(!header_is_opaque_);
  has_dim0_valid_num_ = val;
}

void BlobDesc::set_has_dim1_valid_num_field(bool val) {
  CHECK(!header_is_opaque_);
  has_dim1_valid_num_ = val;
}

void BlobDesc::set_has_dim2_valid_num_field(bool val) {
  CHECK(!header_is_opaque_);
  has_dim2_valid_num_ = val;
}

void BlobDesc::set_has_record_id_in_device_piece_field(bool val) {
  CHECK(!header_is_opaque_);
  has_record_id_in_device_piece_ = val;
}

Shape& BlobDesc::mut_dim0_inner_shape() {
  CHECK(!header_is_opaque_);
  if (!dim0_inner_shape_) { dim0_inner_shape_.reset(new Shape()); }
  return *dim0_inner_shape_;
}

void BlobDesc::DataIdFieldToProto(FieldHeaderDesc* proto, StructPodDesc* header_pod_desc) const {
  UNIMPLEMENTED();
}

void BlobDesc::ColNumFieldToProto(FieldHeaderDesc* proto, StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0)});
  FieldDesc col_num_field(shape, DataType::kInt32);
  col_num_field.ToProto(proto->mutable_col_num());
  header_pod_desc->AddField(FieldKey::kColNum, TensorPodDesc(shape, DataType::kInt32));
}
void BlobDesc::Dim0ValidNumToProto(StructPodDesc* header_pod_desc) const {
  CHECK(dim0_inner_shape_);
  CHECK_EQ(dim0_inner_shape_->elem_cnt(), body_field_.shape().At(0));
  Shape shape({dim0_inner_shape_->At(0)});
  header_pod_desc->AddField(FieldKey::kDim0ValidNum, TensorPodDesc(shape, DataType::kInt64));
}

void BlobDesc::Dim1ValidNumToProto(StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0)});
  header_pod_desc->AddField(FieldKey::kDim1ValidNum, TensorPodDesc(shape, DataType::kInt64));
}

void BlobDesc::Dim2ValidNumToProto(StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0), body_field_.shape().At(1)});
  header_pod_desc->AddField(FieldKey::kDim2ValidNum, TensorPodDesc(shape, DataType::kInt64));
}

void BlobDesc::RecordIdInDevicePieceToProto(StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0)});
  header_pod_desc->AddField(FieldKey::kRecordIdInDevicePiece,
                            TensorPodDesc(shape, DataType::kInt64));
}

void BlobDesc::HeaderToProto(BlobDescProto* proto) const {
  proto->mutable_header()->set_max_col_num(max_col_num_);
  if (!header_is_opaque_) {
    FieldHeaderDesc* field_header = proto->mutable_header()->mutable_field_header();
    StructPodDesc header_pod_desc;
    if (has_data_id_field()) { DataIdFieldToProto(field_header, &header_pod_desc); }
    if (has_col_num_field()) { ColNumFieldToProto(field_header, &header_pod_desc); }
    if (has_dim0_valid_num_field()) { Dim0ValidNumToProto(&header_pod_desc); }
    if (has_dim1_valid_num_field()) { Dim1ValidNumToProto(&header_pod_desc); }
    if (has_dim2_valid_num_field()) { Dim2ValidNumToProto(&header_pod_desc); }
    if (has_record_id_in_device_piece_field()) { RecordIdInDevicePieceToProto(&header_pod_desc); }
    header_pod_desc.ToProto(proto->mutable_header()->mutable_header_pod_desc());
  } else {
    opaque_header_.ToProto(proto->mutable_header()->mutable_opaque_header());
    header_pod_desc_.ToProto(proto->mutable_header()->mutable_header_pod_desc());
  }
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  HeaderToProto(proto);
  body_field_.ToProto(proto->mutable_body());
  if (dim0_inner_shape_) { dim0_inner_shape_->ToProto(proto->mutable_dim0_inner_shape()); }
  proto->set_is_body_disabled(is_body_disabled_);
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return header_is_opaque_ == rhs.header_is_opaque_ && opaque_header_ == rhs.opaque_header_
         && header_pod_desc_ == rhs.header_pod_desc_ && has_data_id_ == rhs.has_data_id_
         && has_col_num_ == rhs.has_col_num_ && has_dim0_valid_num_ == rhs.has_dim0_valid_num_
         && has_dim1_valid_num_ == rhs.has_dim1_valid_num_
         && has_dim2_valid_num_ == rhs.has_dim2_valid_num_
         && has_record_id_in_device_piece_ == rhs.has_record_id_in_device_piece_
         && max_col_num_ == rhs.max_col_num_ && body_field_ == rhs.body_field_
         && is_body_disabled_ == rhs.is_body_disabled_;
}

BlobDesc& BlobDesc::operator=(const BlobDesc& blob_desc) {
  CHECK(blob_desc.is_body_disabled_ == false);  // prevent from misusing
  BlobDescProto proto;
  blob_desc.ToProto(&proto);
  InitFromProto(proto);
  return *this;
}

void BlobDesc::CopyMetaFrom(const BlobDesc& rhs) {
  header_is_opaque_ = rhs.header_is_opaque_;
  opaque_header_ = rhs.opaque_header_;
  header_pod_desc_ = rhs.header_pod_desc_;
  has_data_id_ = rhs.has_data_id_;
  has_col_num_ = rhs.has_col_num_;
  has_dim0_valid_num_ = rhs.has_dim0_valid_num_;
  has_dim1_valid_num_ = rhs.has_dim1_valid_num_;
  has_dim2_valid_num_ = rhs.has_dim2_valid_num_;
  has_record_id_in_device_piece_ = rhs.has_record_id_in_device_piece_;
  max_col_num_ = rhs.max_col_num_;
  body_field_ = rhs.body_field_;
  if (rhs.dim0_inner_shape_) {
    dim0_inner_shape_.reset(new Shape(rhs.dim0_inner_shape_->dim_vec()));
  } else {
    dim0_inner_shape_.reset(nullptr);
  }
}

void BlobDesc::CopyNonMetaFrom(const BlobDesc& rhs) {
  this->is_body_disabled_ = rhs.is_body_disabled_;
}

void BlobDesc::CopyAllFrom(const BlobDesc& rhs) {
  this->CopyMetaFrom(rhs);
  this->CopyNonMetaFrom(rhs);
}

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc) {
  int64_t header_byte_size = 0;
  int64_t body_byte_size = 0;
  HashSet<int> data_type_set;
  int32_t max_col_num = -1;
  int32_t blob_desc_cnt = 0;
  std::unique_ptr<BlobDesc> ret(new BlobDesc(GlobalJobDesc().DefaultDataType()));
  const BlobDesc* last_blob_desc = nullptr;
  StructPodDesc opaque_header_pod_desc;
  bool is_body_disabled = false;
  for (auto& pair : lbi2blob_desc) {
    BlobDesc* blob_desc = pair.second.get();
    RtBlobDesc rt_blob_desc(*blob_desc);
    header_byte_size += rt_blob_desc.ByteSizeOfBlobHeader();
    *opaque_header_pod_desc.MutStructField(NewFieldId(pair.first)) = rt_blob_desc.header_pod_desc();
    body_byte_size += rt_blob_desc.ByteSizeOfBlobBody();
    data_type_set.insert(static_cast<int>(blob_desc->data_type()));
    if (max_col_num == -1) {
      max_col_num = blob_desc->max_col_num();
    } else {
      CHECK_EQ(max_col_num, blob_desc->max_col_num());
    }
    blob_desc_cnt += 1;
    last_blob_desc = blob_desc;
    is_body_disabled = is_body_disabled || blob_desc->is_body_disabled();
  }
  if (blob_desc_cnt == 0) {
    // do nothing
  } else if (blob_desc_cnt == 1) {
    ret.reset(new BlobDesc(*last_blob_desc));
  } else if (data_type_set.size() == 1) {
    CHECK_EQ(false, is_body_disabled);
    DataType sole_data_type = static_cast<DataType>(*(data_type_set.begin()));
    int64_t size_of_one_elem = GetSizeOfDataType(sole_data_type);
    CHECK_EQ(body_byte_size % size_of_one_elem, 0);
    int64_t total_elem_cnt = body_byte_size / size_of_one_elem;
    if (header_byte_size == 0) {
      ret.reset(new BlobDesc(Shape({total_elem_cnt}), sole_data_type, false, false, max_col_num));
    } else {
      ret.reset(new BlobDesc(opaque_header_pod_desc, header_byte_size, Shape({total_elem_cnt}),
                             sole_data_type, max_col_num));
    }
  } else {
    CHECK_EQ(false, is_body_disabled);
    ret.reset(new BlobDesc(opaque_header_pod_desc, header_byte_size, Shape({body_byte_size}),
                           DataType::kChar, max_col_num));
  }
  return ret;
}

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs) {
  return lhs.lbi() < rhs.lbi();
}

}  // namespace oneflow
