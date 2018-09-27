#include "oneflow/core/register/pod_helper.h"

namespace oneflow {

namespace {

size_t SizeOfPod(const PodProto& pod_proto);

size_t SizeOfShapedPod(const ShapedPodProto& shaped_pod) {
  size_t elem_cnt = 1;
  for (int64_t dim : shaped_pod.shape().dim()) {
    CHECK_GE(elem_cnt * dim, elem_cnt);
    elem_cnt *= dim;
  }
  return elem_cnt * GetSizeOfDataType(shaped_pod.data_type());
}

size_t SizeOfNamedField(const NamedField& named_field) {
  return RoundUp(SizeOfPod(named_field.field()), 1 << named_field.align_shift());
}

size_t SizeOfStructPod(const StructPodProto& struct_pod) {
  size_t size = 0;
  for (const NamedField& field : struct_pod.field()) { size += SizeOfNamedField(field); }
  return size;
}

size_t SizeOfPod(const PodProto& pod_proto) {
  if (pod_proto.has_shaped_pod()) { return SizeOfShapedPod(pod_proto.shaped_pod()); }
  if (pod_proto.has_struct_pod()) { return SizeOfStructPod(pod_proto.struct_pod()); }
  UNIMPLEMENTED();
}

}  // namespace

const PodProto& PodHelper::pod_proto() const {
  if (pod_proto_) { return *pod_proto_; }
  if (mut_pod_proto_) { return *mut_pod_proto_; }
  UNIMPLEMENTED();
}

PodProto* PodHelper::mut_pod_proto() { return mut_pod_proto_; }

Shape PodHelper::GetShape() const {
  CHECK(pod_proto().has_shaped_pod());
  return Shape(pod_proto().shaped_pod().shape());
}

DataType PodHelper::GetDataType() const {
  CHECK(pod_proto().has_shaped_pod());
  return pod_proto().shaped_pod().data_type();
}

size_t PodHelper::ByteSize() const { return SizeOfPod(pod_proto()); }

int32_t PodHelper::GetFieldOffset(const std::string& field_name) const {
  CHECK(pod_proto().has_struct_pod());
  const StructPodProto& struct_pod = pod_proto().struct_pod();
  for (int32_t i = 0; i < struct_pod.field_size(); ++i) {
    if (struct_pod.field(i).name() == field_name) { return i; }
  }
  return -1;
}

size_t PodHelper::PtrOffset4Field(int32_t field_idx) const {
  CHECK(pod_proto().has_struct_pod());
  const StructPodProto& struct_pod = pod_proto().struct_pod();
  CHECK_GE(field_idx, 0);
  CHECK_LT(field_idx, struct_pod.field_size());
  size_t offset = 0;
  for (int32_t i = 0; i < field_idx; ++i) { offset += SizeOfNamedField(struct_pod.field(i)); }
  return offset;
}

size_t PodHelper::PtrOffset4Field(const std::string& field_name) const {
  return PtrOffset4Field(GetFieldOffset(field_name));
}

bool PodHelper::HasField(const std::string& field_name) const {
  return pod_proto().has_struct_pod() && GetFieldOffset(field_name) >= 0;
}

PodHelper PodHelper::Field(const std::string& field_name) const {
  CHECK(pod_proto().has_struct_pod());
  return PodHelper(pod_proto().struct_pod().field(GetFieldOffset(field_name)).field());
}

PodHelper PodHelper::MutField(const std::string& field_name) {
  StructPodProto* struct_pod = mut_pod_proto()->mutable_struct_pod();
  int32_t field_offset = GetFieldOffset(field_name);
  NamedField* named_field = nullptr;
  if (field_offset >= 0) {
    named_field = struct_pod->mutable_field(field_offset);
  } else {
    NamedField* named_field = struct_pod->add_field();
    named_field->set_name(field_name);
  }
  return PodHelper(named_field->mutable_field());
}

void PodHelper::SetShapeAndDataType(const Shape& shape, DataType data_type) {
  ShapedPodProto* shaped_pod = mut_pod_proto()->mutable_shaped_pod();
  shape.ToProto(shaped_pod->mutable_shape());
  shaped_pod->set_data_type(data_type);
}

}  // namespace oneflow
