#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc) {
  // TODO(niuchong) : remove PackedBlob
  int64_t header_byte_size = 0;
  int64_t body_byte_size = 0;
  StructPodDesc opaque_header_pod_desc;
  std::unique_ptr<BlobDesc> ret;
  for (const auto& pair : lbi2blob_desc) {
    if (lbi2blob_desc.size() == 1) {
      ret.reset(new BlobDesc(*(pair.second)));
      break;
    }
    RtBlobDesc rt_blob_desc(*(pair.second));
    CHECK_EQ(0, rt_blob_desc.num_of_lod_levels());
    CHECK(!rt_blob_desc.is_body_disabled());
    header_byte_size += rt_blob_desc.ByteSizeOfBlobHeader();
    body_byte_size += rt_blob_desc.AlignedByteSizeOfBlobBody();
    *opaque_header_pod_desc.MutStructField(NewFieldId(pair.first)) = rt_blob_desc.header_pod_desc();
  }
  if (lbi2blob_desc.size() > 1) {
    ret.reset(new BlobDesc(Shape(std::vector<int64_t>{body_byte_size}), DataType::kChar));
    ret->SetOpaqueHeader(opaque_header_pod_desc, header_byte_size);
  }
  return ret;
}

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs) {
  return lhs.lbi() < rhs.lbi();
}

BlobDesc::BlobDesc(const Shape& shape, DataType dtype)
    : body_(shape, dtype),
      header_(),
      num_of_lod_levels_(0),
      is_body_disabled_(false),
      opaque_header_(),
      header_is_opaque_(false) {
  Shape shape_of_dense_shape = Shape({shape.NumAxes()});
  header_.AddField(FieldKey::kDenseShape, TensorPodDesc(shape_of_dense_shape, DataType::kInt64));
}

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
  header_.InitFromProto(proto.header());
  num_of_lod_levels_ = proto.num_of_lod_levels();
  is_body_disabled_ = proto.is_body_disabled();
  opaque_header_.InitFromProto(proto.opaque_header());
  header_is_opaque_ = proto.header_is_opaque();
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  body_.ToProto(proto->mutable_body());
  header_.ToProto(proto->mutable_header());
  proto->set_num_of_lod_levels(num_of_lod_levels_);
  proto->set_is_body_disabled(is_body_disabled_);
  opaque_header_.ToProto(proto->mutable_opaque_header());
  proto->set_header_is_opaque(header_is_opaque_);
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

void BlobDesc::SetLoD(int64_t num_of_lod_levels) {
  CHECK_GT(num_of_lod_levels, 1);
  CHECK_LT(num_of_lod_levels, shape().NumAxes());

  CHECK_GT(shape().NumAxes(), num_of_lod_levels);
  num_of_lod_levels_ = num_of_lod_levels;

  int64_t max_reserved_size_for_lod = 1;
  int64_t cur_level_size = 1;
  for (int64_t i = 0; i < num_of_lod_levels_ - 1; ++i) {
    cur_level_size *= shape().At(i);
    max_reserved_size_for_lod += cur_level_size;
  }

  header_.AddField(
      FieldKey::kLoD,
      TensorPodDesc(Shape(std::vector<int64_t>{max_reserved_size_for_lod}), DataType::kInt64));
  TensorPodDesc* dense_shape_desc =
      header_.MutExistedField(FieldKey::kDenseShape)->MutCast<TensorPodDesc>();
  *(dense_shape_desc->mut_shape()) =
      Shape(std::vector<int64_t>{shape().NumAxes() - num_of_lod_levels});
}

void BlobDesc::SetOpaqueHeader(const StructPodDesc& header_pod_desc, int64_t header_byte_size) {
  CHECK_EQ(num_of_lod_levels_, 0);
  CHECK_GT(header_byte_size, 0);
  CHECK_EQ(header_pod_desc.ByteSize(), header_byte_size);
  header_is_opaque_ = true;
  *opaque_header_.mut_shape() = Shape(std::vector<int64_t>{header_byte_size});
  opaque_header_.set_data_type(DataType::kChar);
  header_ = header_pod_desc;
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  return (body_ == rhs.body_) && (header_ == rhs.header_)
         && (num_of_lod_levels_ == rhs.num_of_lod_levels_)
         && (is_body_disabled_ == rhs.is_body_disabled_) && (opaque_header_ == rhs.opaque_header_)
         && (header_is_opaque_ == rhs.header_is_opaque_);
}

}  // namespace oneflow
