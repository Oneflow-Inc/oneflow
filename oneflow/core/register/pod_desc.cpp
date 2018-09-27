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

void ShapedPodDesc::ToProto(PodProto* pod_proto) const {
  shape_.ToProto(pod_proto->mutable_shaped_pod()->mutable_shape());
  pod_proto->mutable_shaped_pod()->set_data_type(data_type_);
}

AlignedFieldPodDesc::AlignedFieldPodDesc(const AlignedFieldPodProto& aligned_field_pod) {
  name_ = aligned_field_pod.name();
  field_ = std::move(NewPodDesc(aligned_field_pod.field()));
  align_shift_ = aligned_field_pod.align_shift();
}

size_t AlignedFieldPodDesc::ByteSize() const {
  return RoundUp(field_->ByteSize(), 1 << align_shift_);
}

void AlignedFieldPodDesc::ToProto(AlignedFieldPodProto* aligned_field_proto) const {
  aligned_field_proto->set_name(name_);
  aligned_field_proto->set_align_shift(align_shift_);
  field_->ToProto(aligned_field_proto->mutable_field());
}

StructPodDesc::StructPodDesc(const StructPodProto& struct_pod_proto) {
  InitFromProto(struct_pod_proto);
}

StructPodDesc::StructPodDesc(const StructPodDesc& struct_pod_desc) {
  StructPodProto struct_pod_proto;
  struct_pod_desc.ToProto(&struct_pod_proto);
  InitFromProto(struct_pod_proto);
}

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

void StructPodDesc::ToProto(StructPodProto* struct_pod_proto) const {
  for (const auto& field : fields_) { field->ToProto(struct_pod_proto->add_field()); }
}

bool StructPodDesc::HasField(const std::string& name) const {
  return name2field_idx_.find(name) != name2field_idx_.end();
}

const PodDesc& StructPodDesc::Field(const std::string& name) const {
  CHECK(HasField(name));
  return fields_.at(name2field_idx_.at(name))->field();
}

void StructPodDesc::AddCopedField(const std::string& name, const PodDesc& pod_desc,
                                  size_t align_shift) {
  AddField(name, pod_desc.Clone(), align_shift);
}

void StructPodDesc::AddField(const std::string& name, const Shape& shape, DataType data_type,
                             size_t align_shift) {
  AddField(name, std::make_unique<ShapedPodDesc>(shape, data_type), align_shift);
}

void StructPodDesc::AddField(const std::string& name, std::unique_ptr<PodDesc>&& field,
                             size_t align_shift) {
  auto* aligned_pod = new AlignedFieldPodDesc(name, std::move(field), align_shift);
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

}  // namespace oneflow
