#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

std::unique_ptr<BlobDesc> ComputePackedBlobDesc(
    const HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>& lbi2blob_desc) {
  // TODO(niuchong) : remove PackedBlob
  UNIMPLEMENTED();
}

bool CompareLbiBlobDescPair(const LbiBlobDescPair& lhs, const LbiBlobDescPair& rhs) {
  return lhs.lbi() < rhs.lbi();
}

BlobDesc::BlobDesc(const Shape& shape, DataType dtype)
    : body_(shape, dtype), header_(), num_of_lod_levels_(0), is_body_disabled_(false) {
  Shape shape_of_dense_shape = Shape(std::vector{shape.NumAxes()});
  header_.AddField(FieldKey::kDenseShap, TensorPodDesc(shape_of_dense_shape, DataType::kInt64));
}

BlobDesc::BlobDesc(const BlobDescProto& proto) { InitFromProto(proto); }

BlobDesc::BlobDesc(const BlobDesc& other) {
  BlobDescProto proto;
  other->ToProto(&proto);
  InitFromProto(proto);
}

void BlobDesc::InitFromProto(const BlobDescProto& proto) {
  body_.InitFromProto(proto.body());
  header_.InitFromProto(proto.header());
  num_of_lod_levels_ = proto.num_of_lod_levels();
  is_body_disabled_ = proto.is_body_disabled();
}

void BlobDesc::ToProto(BlobDescProto* proto) const {
  body_.ToProto(&(proto->mutable_body()));
  header_.ToProto(&(proto->mutable_header()));
  proto->set_num_of_lod_levels(num_of_lod_levels_);
  proto->set_is_body_disabled(is_body_disabled_);
}

BlobDesc& BlobDesc::operator=(const BlobDesc& rhs) {
  CHECK(rhs.is_body_disabled() == false);  // prevent from misuse
  this->CopyFrom(rhs);
}

void BlobDesc::CopyFrom(const BlobDesc& other) {
  BlobDescProto proto;
  other.ToProto(&proto);
  this->InitFromProto(proto);
}

void BlobDesc::SetLoD(int64_t num_of_lod_levels) {
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
  TensorPodDesc* dense_shape_desc = header_.Field(FieldKey::kDenseShap).MutCast<TensorPodDesc>();
  *(dense_shape_desc->mut_shape()) =
      Shape(std::vector<int64_t>{shape().NumAxes() - num_of_lod_levels});
}

bool BlobDesc::operator==(const BlobDesc& rhs) const {
  BlobDescProto lhs_proto;
  this->ToProto(&lhs_proto);
  BlobDescProto rhs_proto;
  this->ToProto(&lhs_proto);
  return lhs_proto == rhs_proto;
}

void BlobDesc::set_num_of_lod_levels(int64_t val) {
  CHECK_GT(val, 1);
  CHECK_LT(val, shape().NumAxes());
  num_of_lod_levels_ = val;
}

}  // namespace oneflow
