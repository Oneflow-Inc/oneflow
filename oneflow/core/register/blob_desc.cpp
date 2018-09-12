#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

BlobDesc::BlobDesc()
    : BlobDesc(Shape(), Global<JobDesc>::Get()->DefaultDataType(), false, false, false, 1) {}

BlobDesc::BlobDesc(const Shape& shape, DataType data_type, bool has_data_id, bool has_col_num,
                   bool has_instance_num, int32_t max_col_num)
    : header_is_opaque_(false),
      has_data_id_(has_data_id),
      has_col_num_(has_col_num),
      has_instance_num_(has_instance_num),
      max_col_num_(max_col_num),
      blob_mem_id_(-1),
      body_field_(shape, data_type) {}

BlobDesc::BlobDesc(const BlobDescProto& proto) : body_field_(proto.body()) {
  max_col_num_ = proto.header().max_col_num();
  blob_mem_id_ = proto.header().blob_mem_id();
  if (proto.header().has_opaque_header()) {
    header_is_opaque_ = true;
    has_data_id_ = false;
    has_col_num_ = false;
    has_instance_num_ = false;
    opaque_header_ = FieldDesc(proto.header().opaque_header());
  } else {
    CHECK(proto.header().has_field_header());
    header_is_opaque_ = false;
    has_data_id_ = proto.header().field_header().has_data_id();
    has_col_num_ = proto.header().field_header().has_col_num();
    has_instance_num_ = proto.header().field_header().has_instance_num();
  }
}

BlobDesc::BlobDesc(int64_t header_byte_size, const Shape& shape, DataType data_type,
                   int32_t max_col_num)
    : has_data_id_(false),
      has_col_num_(false),
      has_instance_num_(false),
      max_col_num_(max_col_num),
      blob_mem_id_(-1),
      body_field_(shape, data_type) {
  if (header_byte_size > 0) {
    header_is_opaque_ = true;
    opaque_header_ = FieldDesc(Shape({header_byte_size}), DataType::kChar);
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

void BlobDesc::set_has_instance_num_field(bool val) {
  CHECK(!header_is_opaque_);
  has_instance_num_ = val;
}

void BlobDesc::DataIdFieldToProto(FieldHeaderDesc* proto) const {
  FieldDesc data_id_field(Shape({body_field_.shape().At(0),
                                 static_cast<int64_t>(Global<JobDesc>::Get()->SizeOfOneDataId())}),
                          DataType::kChar);
  data_id_field.ToProto(proto->mutable_data_id());
}

void BlobDesc::ColNumFieldToProto(FieldHeaderDesc* proto) const {
  FieldDesc col_num_field(Shape({body_field_.shape().At(0)}), DataType::kInt32);
  col_num_field.ToProto(proto->mutable_col_num());
}

void BlobDesc::InstanceNumFieldToProto(FieldHeaderDesc* proto) const {
  FieldDesc instance_num_field((Shape({1})), DataType::kInt32);
  instance_num_field.ToProto(proto->mutable_instance_num());
}

void BlobDesc::HeaderToProto(BlobDescProto* proto) const {
  proto->mutable_header()->set_max_col_num(max_col_num_);
  proto->mutable_header()->set_blob_mem_id(blob_mem_id_);
  if (!header_is_opaque_) {
    FieldHeaderDesc* field_header = proto->mutable_header()->mutable_field_header();
    if (has_data_id_field()) { DataIdFieldToProto(field_header); }
    if (has_col_num_field()) { ColNumFieldToProto(field_header); }
    if (has_instance_num_field()) { InstanceNumFieldToProto(field_header); }
  } else {
    opaque_header_.ToProto(proto->mutable_header()->mutable_opaque_header());
  }
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  HeaderToProto(proto);
  body_field_.ToProto(proto->mutable_body());
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return header_is_opaque_ == rhs.header_is_opaque_ && opaque_header_ == rhs.opaque_header_
         && has_data_id_ == rhs.has_data_id_ && has_col_num_ == rhs.has_col_num_
         && has_instance_num_ == rhs.has_instance_num_ && max_col_num_ == rhs.max_col_num_
         && blob_mem_id_ == rhs.blob_mem_id_ && body_field_ == rhs.body_field_;
}

BlobDesc& BlobDesc::operator=(const BlobDesc& blob_desc) {
  header_is_opaque_ = blob_desc.header_is_opaque_;
  opaque_header_ = blob_desc.opaque_header_;
  has_data_id_ = blob_desc.has_data_id_;
  has_col_num_ = blob_desc.has_col_num_;
  has_instance_num_ = blob_desc.has_instance_num_;
  max_col_num_ = blob_desc.max_col_num_;
  body_field_ = blob_desc.body_field_;
  blob_mem_id_ = -1;
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

  for (auto& pair : lbi2blob_desc) {
    BlobDesc* blob_desc = pair.second.get();
    RtBlobDesc rt_blob_desc(*blob_desc);
    header_byte_size += rt_blob_desc.ByteSizeOfBlobHeader();
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
      ret.reset(
          new BlobDesc(Shape({total_elem_cnt}), sole_data_type, false, false, false, max_col_num));
    } else {
      ret.reset(
          new BlobDesc(header_byte_size, Shape({total_elem_cnt}), sole_data_type, max_col_num));
    }
  } else {
    ret.reset(
        new BlobDesc(header_byte_size, Shape({body_byte_size}), DataType::kChar, max_col_num));
  }
  return ret;
}

}  // namespace oneflow
