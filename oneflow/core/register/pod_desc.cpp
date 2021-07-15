/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/register/pod_desc.h"

namespace oneflow {

namespace {

std::unique_ptr<PodDesc> NewPodDesc(const PodProto& pod) {
  if (pod.has_tensor_pod()) { return std::make_unique<TensorPodDesc>(pod.tensor_pod()); }
  if (pod.has_struct_pod()) { return std::make_unique<StructPodDesc>(pod.struct_pod()); }
  // ignore field pod
  UNIMPLEMENTED();
  return std::unique_ptr<PodDesc>();
}

}  // namespace

FieldId NewFieldId(FieldKey key) {
  FieldId ret;
  ret.set_key(key);
  return ret;
}

FieldId NewFieldId(const LogicalBlobId& lbi) {
  FieldId ret;
  *ret.mutable_lbi() = lbi;
  return ret;
}

TensorPodDesc::TensorPodDesc() : PodDesc(), shape_(std::make_shared<Shape>()) {}
TensorPodDesc::TensorPodDesc(const Shape& shape, DataType data_type)
    : PodDesc(), shape_(std::make_shared<Shape>(shape)), data_type_(data_type) {}
TensorPodDesc::TensorPodDesc(const std::shared_ptr<Shape>& shape, DataType data_type)
    : PodDesc(), shape_(shape), data_type_(data_type) {}

TensorPodDesc::TensorPodDesc(const TensorPodProto& tensor_pod) {
  shape_ = std::make_shared<Shape>();
  InitFromProto(tensor_pod);
}

TensorPodDesc::TensorPodDesc(const TensorPodDesc& tensor_pod) {
  shape_ = std::make_shared<Shape>();
  PodProto pod_proto;
  tensor_pod.ToProto(&pod_proto);
  InitFromProto(pod_proto.tensor_pod());
}

void TensorPodDesc::InitFromProto(const TensorPodProto& tensor_pod) {
  *mut_shape() = Shape(tensor_pod.shape());
  data_type_ = tensor_pod.data_type();
}

size_t TensorPodDesc::ByteSize() const {
  return shape().elem_cnt() * GetSizeOfDataType(data_type_);
}

bool TensorPodDesc::operator==(const PodDesc& rhs) const {
  const auto* tensor_rhs = dynamic_cast<const TensorPodDesc*>(&rhs);
  if (tensor_rhs == nullptr) { return false; }
  return shape() == tensor_rhs->shape() && data_type() == tensor_rhs->data_type();
}

void TensorPodDesc::ToProto(PodProto* pod_proto) const { ToProto(pod_proto->mutable_tensor_pod()); }

void TensorPodDesc::ToProto(TensorPodProto* proto) const {
  shape().ToProto(proto->mutable_shape());
  proto->set_data_type(data_type_);
}

FieldPodDesc::FieldPodDesc(const FieldPodProto& field_pod) {
  field_id_ = field_pod.field_id();
  pod_ = std::move(NewPodDesc(field_pod.pod()));
  alignment_ = field_pod.alignment();
}

size_t FieldPodDesc::ByteSize() const { return RoundUp(pod_->ByteSize(), alignment_); }

bool FieldPodDesc::operator==(const PodDesc& rhs) const {
  const auto* field_rhs = dynamic_cast<const FieldPodDesc*>(&rhs);
  if (field_rhs == nullptr) { return false; }
  return field_id() == field_rhs->field_id() && pod() == field_rhs->pod()
         && alignment_ == field_rhs->alignment_;
}

void FieldPodDesc::ToProto(FieldPodProto* field_pod_proto) const {
  *field_pod_proto->mutable_field_id() = field_id_;
  field_pod_proto->set_alignment(alignment_);
  pod_->ToProto(field_pod_proto->mutable_pod());
}

StructPodDesc::StructPodDesc(const StructPodProto& struct_pod_proto) {
  InitFromProto(struct_pod_proto);
}

StructPodDesc::StructPodDesc(const StructPodDesc& struct_pod_desc) { *this = struct_pod_desc; }

void StructPodDesc::InitFromProto(const StructPodProto& struct_pod) {
  Clear();
  for (const auto& field : struct_pod.field()) {
    std::unique_ptr<FieldPodDesc> pod(new FieldPodDesc(field));
    AddField(std::move(pod));
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
  if (field_id2field_idx_ != struct_rhs->field_id2field_idx_) { return false; }
  for (int i = 0; i < field_id2field_idx_.size(); ++i) {
    if (*fields_.at(i) != *struct_rhs->fields_.at(i)) { return false; }
  }
  return true;
}

void StructPodDesc::ToProto(StructPodProto* struct_pod_proto) const {
  struct_pod_proto->Clear();
  for (const auto& field : fields_) { field->ToProto(struct_pod_proto->add_field()); }
}

bool StructPodDesc::HasField(const FieldId& field_id) const {
  return field_id2field_idx_.find(field_id) != field_id2field_idx_.end();
}

StructPodDesc* StructPodDesc::MutStructField(const FieldId& field_id) {
  return MutStructField(field_id, 1);
}

StructPodDesc* StructPodDesc::MutStructField(const FieldId& field_id, int32_t alignment) {
  if (!HasField(field_id)) { AddField(field_id, std::make_unique<StructPodDesc>(), alignment); }
  return MutExistedField(field_id)->MutCast<StructPodDesc>();
}

PodDesc* StructPodDesc::MutExistedField(const FieldId& field_id) {
  return fields_.at(field_id2field_idx_.at(field_id))->mut_pod();
}

const PodDesc& StructPodDesc::Field(const FieldId& field_id) const {
  return fields_.at(field_id2field_idx_.at(field_id))->pod();
}

void StructPodDesc::AddField(FieldKey field_key, const PodDesc& pod_desc) {
  return AddField(NewFieldId(field_key), pod_desc);
}

void StructPodDesc::AddField(const FieldId& field_id, const PodDesc& pod_desc) {
  return AddField(field_id, pod_desc, 1);
}

void StructPodDesc::AddField(const FieldId& field_id, const PodDesc& pod_desc, size_t alignment) {
  AddField(field_id, pod_desc.Clone(), alignment);
}

void StructPodDesc::AddField(const FieldId& field_id, std::unique_ptr<PodDesc>&& field,
                             size_t alignment) {
  auto* pod = new FieldPodDesc(field_id, std::move(field), alignment);
  AddField(std::unique_ptr<FieldPodDesc>(pod));
}

void StructPodDesc::AddField(std::unique_ptr<FieldPodDesc>&& field) {
  CHECK(field_id2field_idx_.emplace(field->field_id(), fields_.size()).second);
  fields_.emplace_back(std::move(field));
}

size_t StructPodDesc::ByteOffset4Field(const FieldId& field_id) const {
  CHECK(HasField(field_id));
  size_t offset = 0;
  for (int32_t i = 0; i < field_id2field_idx_.at(field_id); ++i) {
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
  CHECK_EQ(fields_.size(), field_id2field_idx_.size());
  fields_.clear();
  field_id2field_idx_.clear();
}

}  // namespace oneflow
