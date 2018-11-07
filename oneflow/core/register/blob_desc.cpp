#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

#define INIT_HAS_FIELD_FALSE(field_key, field_name) has_##field_name##_ = false;
#define INIT_HAS_FIELD_BY_POD(field_key, field_name) \
  has_##field_name##_ = header_pod_desc_.HasField(field_key);

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType(), false, false, 1) {}

BlobDesc::BlobDesc(const Shape& shape, DataType data_type, bool has_data_id, bool has_col_num,
                   int32_t max_col_num)
    : header_is_opaque_(false),
      has_data_id_(has_data_id),
      has_col_num_(has_col_num),
      has_dim0_valid_num_(false),
      has_dim1_valid_num_(false),
      has_dim2_valid_num_(false),
      has_record_id_in_device_piece_(false),
      max_col_num_(max_col_num),
      blob_mem_id_(-1),
      body_field_(shape, data_type) {}

BlobDesc::BlobDesc(const BlobDesc& BlobDesc) { *this = BlobDesc; }

void BlobDesc::InitFromProto(const BlobDescProto& proto) {
  body_field_.InitFromProto(proto.body());
  max_col_num_ = proto.header().max_col_num();
  blob_mem_id_ = proto.header().blob_mem_id();
  header_pod_desc_.InitFromProto(proto.header().header_pod_desc());
  if (proto.header().has_opaque_header()) {
    header_is_opaque_ = true;
    OF_PP_FOR_EACH_TUPLE(INIT_HAS_FIELD_FALSE, FIELD_KEY_AND_FIELD_NAME_SEQ)
    opaque_header_ = FieldDesc(proto.header().opaque_header());
  } else {
    CHECK(proto.header().has_field_header());
    header_is_opaque_ = false;
    OF_PP_FOR_EACH_TUPLE(INIT_HAS_FIELD_BY_POD, FIELD_KEY_AND_FIELD_NAME_SEQ)
  }
  if (proto.has_dim0_inner_shape()) {
    dim0_inner_shape_.reset(new Shape(proto.dim0_inner_shape()));
  }
}

BlobDesc::BlobDesc(const StructPodDesc& header_pod_desc, int64_t header_byte_size,
                   const Shape& shape, DataType data_type, int32_t max_col_num)
    : max_col_num_(max_col_num), blob_mem_id_(-1), body_field_(shape, data_type) {
  OF_PP_FOR_EACH_TUPLE(INIT_HAS_FIELD_FALSE, FIELD_KEY_AND_FIELD_NAME_SEQ)
  CHECK_EQ(header_pod_desc.ByteSize(), header_byte_size);
  if (header_byte_size > 0) {
    header_is_opaque_ = true;
    opaque_header_ = FieldDesc(Shape({header_byte_size}), DataType::kChar);
    header_pod_desc_ = header_pod_desc;
  } else {
    header_is_opaque_ = false;
  }
}

#define DEFINE_HAS_FIELD_GETTER_AND_SETTER(field_key, field_name) \
  template<>                                                      \
  bool BlobDesc::HasField<field_key>() const {                    \
    return has_##field_name##_;                                   \
  }                                                               \
  template<>                                                      \
  void BlobDesc::SetHasField<field_key>(bool val) {               \
    CHECK(!header_is_opaque_);                                    \
    has_##field_name##_ = val;                                    \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_HAS_FIELD_GETTER_AND_SETTER, FIELD_KEY_AND_FIELD_NAME_SEQ)

Shape& BlobDesc::mut_dim0_inner_shape() {
  CHECK(!header_is_opaque_);
  if (!dim0_inner_shape_) { dim0_inner_shape_.reset(new Shape()); }
  return *dim0_inner_shape_;
}

template<>
void BlobDesc::FieldToProto<FieldKey::kDataId>(FieldHeaderDesc* proto,
                                               StructPodDesc* header_pod_desc) const {
  Shape shape(
      {body_field_.shape().At(0), static_cast<int64_t>(Global<JobDesc>::Get()->SizeOfOneDataId())});
  FieldDesc data_id_field(shape, DataType::kChar);
  data_id_field.ToProto(proto->mutable_data_id());
  header_pod_desc->AddField(FieldKey::kDataId, TensorPodDesc(shape, DataType::kChar));
}
template<>
void BlobDesc::FieldToProto<FieldKey::kColNum>(FieldHeaderDesc* proto,
                                               StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0)});
  FieldDesc col_num_field(shape, DataType::kInt32);
  col_num_field.ToProto(proto->mutable_col_num());
  header_pod_desc->AddField(FieldKey::kColNum, TensorPodDesc(shape, DataType::kInt32));
}
template<>
void BlobDesc::FieldToProto<FieldKey::kDim0ValidNum>(FieldHeaderDesc* proto,
                                                     StructPodDesc* header_pod_desc) const {
  CHECK(dim0_inner_shape_);
  CHECK_EQ(dim0_inner_shape_->elem_cnt(), body_field_.shape().At(0));
  Shape shape({dim0_inner_shape_->At(0)});
  header_pod_desc->AddField(FieldKey::kDim0ValidNum, TensorPodDesc(shape, DataType::kInt64));
}
template<>
void BlobDesc::FieldToProto<FieldKey::kDim1ValidNum>(FieldHeaderDesc* proto,
                                                     StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0)});
  header_pod_desc->AddField(FieldKey::kDim1ValidNum, TensorPodDesc(shape, DataType::kInt64));
}
template<>
void BlobDesc::FieldToProto<FieldKey::kDim2ValidNum>(FieldHeaderDesc* proto,
                                                     StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0), body_field_.shape().At(1)});
  header_pod_desc->AddField(FieldKey::kDim2ValidNum, TensorPodDesc(shape, DataType::kInt64));
}
template<>
void BlobDesc::FieldToProto<FieldKey::kRecordIdInDevicePiece>(
    FieldHeaderDesc* proto, StructPodDesc* header_pod_desc) const {
  Shape shape({body_field_.shape().At(0)});
  header_pod_desc->AddField(FieldKey::kRecordIdInDevicePiece,
                            TensorPodDesc(shape, DataType::kInt64));
}

void BlobDesc::HeaderToProto(BlobDescProto* proto) const {
  proto->mutable_header()->set_max_col_num(max_col_num_);
  proto->mutable_header()->set_blob_mem_id(blob_mem_id_);
  if (!header_is_opaque_) {
    FieldHeaderDesc* field_header = proto->mutable_header()->mutable_field_header();
    StructPodDesc header_pod_desc;
#define HAS_FIELD_THEN_TO_PROTO(field_key, field_name) \
  if (HasField<field_key>()) { FieldToProto<field_key>(field_header, &header_pod_desc); }
    OF_PP_FOR_EACH_TUPLE(HAS_FIELD_THEN_TO_PROTO, FIELD_KEY_AND_FIELD_NAME_SEQ)
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
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
#define HAS_FIELD_EQUAL(field_key, field_name) &&has_##field_name##_ == rhs.has_##field_name##_
  return header_is_opaque_ == rhs.header_is_opaque_ && opaque_header_ == rhs.opaque_header_
         && header_pod_desc_ == rhs.header_pod_desc_ && max_col_num_ == rhs.max_col_num_
         && blob_mem_id_ == rhs.blob_mem_id_
         && body_field_
                == rhs.body_field_ OF_PP_FOR_EACH_TUPLE(HAS_FIELD_EQUAL,
                                                        FIELD_KEY_AND_FIELD_NAME_SEQ);
}

BlobDesc& BlobDesc::operator=(const BlobDesc& blob_desc) {
  BlobDescProto proto;
  blob_desc.ToProto(&proto);
  InitFromProto(proto);
  return *this;
}

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc) {
  int64_t header_byte_size = 0;
  int64_t body_byte_size = 0;
  HashSet<int> data_type_set;
  int32_t max_col_num = -1;
  int32_t blob_desc_cnt = 0;
  std::unique_ptr<BlobDesc> ret(new BlobDesc());
  const BlobDesc* last_blob_desc = nullptr;
  HashMap<int32_t, size_t> blob_mem_id2size;
  StructPodDesc opaque_header_pod_desc;
  for (auto& pair : lbi2blob_desc) {
    BlobDesc* blob_desc = pair.second.get();
    RtBlobDesc rt_blob_desc(*blob_desc);
    header_byte_size += rt_blob_desc.ByteSizeOfBlobHeader();
    *opaque_header_pod_desc.MutStructField(NewFieldId(pair.first)) = rt_blob_desc.header_pod_desc();
    int64_t cur_body_byte_size = rt_blob_desc.ByteSizeOfBlobBody();
    int32_t blob_mem_id = blob_desc->blob_mem_id();
    if (blob_mem_id == -1) {
      body_byte_size += cur_body_byte_size;
    } else {
      auto size_it = blob_mem_id2size.find(blob_mem_id);
      if (size_it == blob_mem_id2size.end()) {
        CHECK(blob_mem_id2size.emplace(blob_mem_id, cur_body_byte_size).second);
      } else {
        CHECK_EQ(size_it->second, cur_body_byte_size);
      }
    }
    data_type_set.insert(static_cast<int>(blob_desc->data_type()));
    if (max_col_num == -1) {
      max_col_num = blob_desc->max_col_num();
    } else {
      CHECK_EQ(max_col_num, blob_desc->max_col_num());
    }
    blob_desc_cnt += 1;
    last_blob_desc = blob_desc;
  }
  for (auto& pair : blob_mem_id2size) { body_byte_size += pair.second; }
  if (blob_desc_cnt == 0) {
    // do nothing
  } else if (blob_desc_cnt == 1) {
    ret.reset(new BlobDesc(*last_blob_desc));
  } else if (data_type_set.size() == 1) {
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
    ret.reset(new BlobDesc(opaque_header_pod_desc, header_byte_size, Shape({body_byte_size}),
                           DataType::kChar, max_col_num));
  }
  return ret;
}

}  // namespace oneflow
