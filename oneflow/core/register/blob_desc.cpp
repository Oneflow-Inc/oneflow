#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc) {
  // TODO(niuchong) : remove PackedBlob
  int64_t body_byte_size = 0;
  StructPodDesc opaque_header_pod_desc;
  std::unique_ptr<BlobDesc> ret;
  for (const auto& pair : lbi2blob_desc) {
    if (lbi2blob_desc.size() == 1) {
      ret.reset(new BlobDesc(*(pair.second)));
      break;
    }
    RtBlobDesc rt_blob_desc(*(pair.second));
    // CHECK(!rt_blob_desc.is_dynamic());
    CHECK(!rt_blob_desc.is_body_disabled());
    body_byte_size += rt_blob_desc.AlignedByteSizeOfBlobBody();
    *opaque_header_pod_desc.MutStructField(NewFieldId(pair.first)) = rt_blob_desc.header_pod_desc();
  }
  if (lbi2blob_desc.size() > 1) {
    ret.reset(new BlobDesc(Shape(DimVector{body_byte_size}), DataType::kChar));
    ret->SetOpaqueHeader(opaque_header_pod_desc);
  }
  return ret;
}

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs) {
  return lhs.lbi() < rhs.lbi();
}

BlobDesc::BlobDesc(const Shape& shape, DataType dtype)
    : body_(shape, dtype),
      num_of_lod_levels_(0),
      is_body_disabled_(false),
      is_dynamic_(false),
      opaque_header_() {}

BlobDesc::BlobDesc(const BlobDescProto& proto) { InitFromProto(proto); }

BlobDesc::BlobDesc(const BlobDesc& other) {
  // *body_.mut_shape() = other.body_.shape();
  // body_.set_data_type(other.body_.data_type());
  // header_ = other.header_;
  // num_of_lod_levels_ = other.num_of_lod_levels_;
  // is_body_disabled_ = other.is_body_disabled_;
  BlobDescProto proto;
  other.ToProto(&proto);
  InitFromProto(proto);
}

void BlobDesc::InitFromProto(const BlobDescProto& proto) {
  body_.InitFromProto(proto.body());
  num_of_lod_levels_ = proto.num_of_lod_levels();
  is_body_disabled_ = proto.is_body_disabled();
  is_dynamic_ = proto.is_dynamic();
  if (proto.header_is_opaque()) {
    opaque_header_.reset(new StructPodDesc(proto.header()));
  } else {
    opaque_header_.reset(nullptr);
  }
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  body_.ToProto(proto->mutable_body());
  proto->set_num_of_lod_levels(num_of_lod_levels_);
  proto->set_is_body_disabled(is_body_disabled_);
  proto->set_is_dynamic(is_dynamic_);

  if (opaque_header_) {
    opaque_header_->ToProto(proto->mutable_header());
    proto->set_header_is_opaque(true);
  } else {
    StructPodDesc header;
    int64_t dense_shape_num_axes = shape().NumAxes();
    if (num_of_lod_levels_ > 0) {
      CHECK(is_dynamic_);
      int64_t max_reserved_size_for_lod = 1;
      int64_t cur_level_size = 1;
      for (int64_t i = 0; i < num_of_lod_levels_ - 1; ++i) {
        cur_level_size *= shape().At(i);
        max_reserved_size_for_lod += cur_level_size;
      }
      max_reserved_size_for_lod += num_of_lod_levels_;
      header.AddField(
          FieldKey::kLoD,
          TensorPodDesc(Shape(DimVector{max_reserved_size_for_lod + num_of_lod_levels_}),
                        DataType::kInt64));
      dense_shape_num_axes = shape().NumAxes() - num_of_lod_levels_ + 1;  // 1 for tiled lod dims
    }
    header.AddField(FieldKey::kDenseShape,
                    TensorPodDesc(Shape(DimVector{dense_shape_num_axes}), DataType::kInt64));
    header.ToProto(proto->mutable_header());
    proto->set_header_is_opaque(false);
  }
}

BlobDesc& BlobDesc::operator=(const BlobDesc& rhs) {
  CHECK(rhs.is_body_disabled() == false);  // prevent from misuse
  this->CopyFrom(rhs);
  return *this;
}

void BlobDesc::CopyFrom(const BlobDesc& other) {
  BlobDescProto proto;
  other.ToProto(&proto);
  this->InitFromProto(proto);
}

// TODO(niuchong) : remove is_body_disabled from blob into register
void BlobDesc::CopyMetaFrom(const BlobDesc& other) {
  bool tmp = is_body_disabled_;
  CopyFrom(other);
  is_body_disabled_ = tmp;
}

Maybe<void> BlobDesc::SetLoD(int64_t num_of_lod_levels) {
  if (num_of_lod_levels == 0) { return Maybe<void>::Ok(); }
  OF_CHECK_GT(num_of_lod_levels, 1);
  OF_CHECK_LE(num_of_lod_levels, shape().NumAxes());
  num_of_lod_levels_ = num_of_lod_levels;
  is_dynamic_ = true;
  return Maybe<void>::Ok();
}

void BlobDesc::SetOpaqueHeader(const StructPodDesc& header_pod_desc) {
  CHECK(!is_dynamic_);
  CHECK_EQ(num_of_lod_levels_, 0);
  CHECK_GT(header_pod_desc.ByteSize(), 0);
  opaque_header_.reset(new StructPodDesc(header_pod_desc));
}

void BlobDesc::set_is_dynamic(bool is_dynamic) {
  if (!is_dynamic) { CHECK_EQ(0, num_of_lod_levels_); }
  is_dynamic_ = is_dynamic;
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return (body_ == rhs.body_) && (num_of_lod_levels_ == rhs.num_of_lod_levels_)
         && (is_body_disabled_ == rhs.is_body_disabled_) && (is_dynamic_ == rhs.is_dynamic_)
         && (opaque_header_ == rhs.opaque_header_);
}

}  // namespace oneflow
