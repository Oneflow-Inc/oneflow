#include "oneflow/core/register/pod_desc.h"

namespace oneflow {

namespace {

std::unique_ptr<PodDesc> NewPodDesc(const PodProto& pod) {
  if (pod.has_shaped_pod()) { return std::make_unique<ShapedPodDesc>(pod.shaped_pod()); }
  if (pod.has_struct_pod()) { return std::make_unique<StructPodDesc>(pod.struct_pod()); }
  // ignore aligned field pod
  UNIMPLEMENTED();
}

}  // namespace

ShapedPodDesc::ShapedPodDesc(const ShapedPodProto& shaped_pod) : PodDesc() {
  shape_ = Shape(shaped_pod.shape());
  data_type_ = shaped_pod.data_type();
}

size_t ShapedPodDesc::ByteSize() const { return shape_.elem_cnt() * GetSizeOfDataType(data_type_); }

bool ShapedPodDesc::operator==(const PodDesc& rhs) const {
  const auto* shaped_rhs = dynamic_cast<const ShapedPodDesc*>(&rhs);
  if (shaped_rhs == nullptr) { return false; }
  return shape() == shaped_rhs->shape() && data_type() == shaped_rhs->data_type();
}

void ShapedPodDesc::ToProto(PodProto* pod_proto) const {
  shape_.ToProto(pod_proto->mutable_shaped_pod()->mutable_shape());
  pod_proto->mutable_shaped_pod()->set_data_type(data_type_);
}

AlignedFieldPodDesc::AlignedFieldPodDesc(const AlignedFieldPodProto& aligned_field_pod) {
  name_ = aligned_field_pod.name();
  field_ = std::move(NewPodDesc(aligned_field_pod.field()));
  alignment_ = aligned_field_pod.alignment();
}

size_t AlignedFieldPodDesc::ByteSize() const { return RoundUp(field_->ByteSize(), alignment_); }

bool AlignedFieldPodDesc::operator==(const PodDesc& rhs) const {
  const auto* aligned_field_rhs = dynamic_cast<const AlignedFieldPodDesc*>(&rhs);
  if (aligned_field_rhs == nullptr) { return false; }
  return name() == aligned_field_rhs->name() && field() == aligned_field_rhs->field()
         && alignment_ == aligned_field_rhs->alignment_;
}

void AlignedFieldPodDesc::ToProto(AlignedFieldPodProto* aligned_field_proto) const {
  aligned_field_proto->set_name(name_);
  aligned_field_proto->set_alignment(alignment_);
  field_->ToProto(aligned_field_proto->mutable_field());
}

StructPodDesc::StructPodDesc(const StructPodProto& struct_pod_proto) {
  InitFromProto(struct_pod_proto);
}

StructPodDesc::StructPodDesc(const StructPodDesc& struct_pod_desc) { *this = struct_pod_desc; }

void StructPodDesc::InitFromProto(const StructPodProto& struct_pod) {
  CHECK(name2field_idx_.empty());
  CHECK(fields_.empty());
  for (const auto& field : struct_pod.field()) {
    std::unique_ptr<AlignedFieldPodDesc> aligned_pod(new AlignedFieldPodDesc(field));
    AddField(std::move(aligned_pod));
  }
}

size_t StructPodDesc::ByteSize() const {
  size_t size = 0;
  for (const auto& field : fields_) { size += field->ByteSize(); }
  return size;
}

bool StructPodDesc::operator==(const PodDesc& rhs) const {
  const auto* struct_rhs = dynamic_cast<const StructPodDesc*>(&rhs);
  if (struct_rhs == nullptr) { return false; }
  if (name2field_idx_ != struct_rhs->name2field_idx_) { return false; }
  for (int i = 0; i < name2field_idx_.size(); ++i) {
    if (*fields_.at(i) != *struct_rhs->fields_.at(i)) { return false; }
  }
  return true;
}

void StructPodDesc::ToProto(StructPodProto* struct_pod_proto) const {
  for (const auto& field : fields_) { field->ToProto(struct_pod_proto->add_field()); }
}

bool StructPodDesc::HasField(const std::string& name) const {
  return name2field_idx_.find(name) != name2field_idx_.end();
}

StructPodDesc* StructPodDesc::MutStructField(const std::string& name) {
  if (!HasField(name)) { AddField(name, std::make_unique<StructPodDesc>()); }
  return MutExistedField(name)->MutCast<StructPodDesc>();
}

ShapedPodDesc* StructPodDesc::MutShapedField(const std::string& name) {
  if (!HasField(name)) { AddField(name, std::make_unique<ShapedPodDesc>()); }
  return MutExistedField(name)->MutCast<ShapedPodDesc>();
}

PodDesc* StructPodDesc::MutExistedField(const std::string& name) {
  return fields_.at(name2field_idx_.at(name))->mut_field();
}

const PodDesc& StructPodDesc::Field(const std::string& name) const {
  return fields_.at(name2field_idx_.at(name))->field();
}

void StructPodDesc::AddField(const std::string& name, const PodDesc& pod_desc, size_t alignment) {
  AddField(name, pod_desc.Clone(), alignment);
}

void StructPodDesc::AddField(const std::string& name, std::unique_ptr<PodDesc>&& field,
                             size_t alignment) {
  auto* aligned_pod = new AlignedFieldPodDesc(name, std::move(field), alignment);
  AddField(std::unique_ptr<AlignedFieldPodDesc>(aligned_pod));
}

void StructPodDesc::AddField(std::unique_ptr<AlignedFieldPodDesc>&& field) {
  CHECK(name2field_idx_.emplace(field->name(), fields_.size()).second);
  fields_.emplace_back(std::move(field));
}

size_t StructPodDesc::PtrOffset4Field(const std::string& field_name) const {
  CHECK(HasField(field_name));
  size_t offset = 0;
  for (int32_t i = 0; i < name2field_idx_.at(field_name); ++i) {
    offset += fields_.at(i)->ByteSize();
  }
  return offset;
}

StructPodDesc& StructPodDesc::operator=(const StructPodDesc& struct_pod_desc) {
  Clear();
  StructPodProto struct_pod_proto;
  struct_pod_desc.ToProto(&struct_pod_proto);
  InitFromProto(struct_pod_proto);
  return *this;
}

void StructPodDesc::Clear() {
  CHECK_EQ(fields_.size(), name2field_idx_.size());
  fields_.clear();
  name2field_idx_.clear();
}

}  // namespace oneflow
