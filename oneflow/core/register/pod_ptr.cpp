#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

namespace {

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

int32_t FieldOffset4StructPod(const StructPodProto& struct_pod, const std::string& field_name) {
  for (int32_t i = 0; i < struct_pod.field_size(); ++i) {
    if (struct_pod.field(i).name() == field_name) { return i; }
  }
  return -1;
}

size_t PtrOffset4Field(const StructPodProto& struct_pod, int32_t field_idx) {
  CHECK_GE(field_idx, 0);
  CHECK_LT(field_idx, struct_pod.field_size());
  size_t offset = 0;
  for (int32_t i = 0; i < field_idx; ++i) { offset += SizeOfNamedField(struct_pod.field(i)); }
  return offset;
}

}  // namespace

size_t SizeOfPod(const PodProto& pod_proto) {
  if (pod_proto.has_shaped_pod()) { return SizeOfShapedPod(pod_proto.shaped_pod()); }
  if (pod_proto.has_struct_pod()) { return SizeOfStructPod(pod_proto.struct_pod()); }
  UNIMPLEMENTED();
}

bool PodPtr::HasField(const std::string& field_name) const {
  return pod_proto_->has_struct_pod()
         && FieldOffset4StructPod(pod_proto_->struct_pod(), field_name) >= 0;
}

PodPtr PodPtr::Field(const std::string& field_name) const {
  CHECK(pod_proto_->has_struct_pod());
  const StructPodProto& struct_pod = pod_proto_->struct_pod();
  int32_t field_offset = FieldOffset4StructPod(struct_pod, field_name);
  CHECK_GE(field_offset, 0);
  size_t ptr_offset = PtrOffset4Field(struct_pod, field_offset);
  return PodPtr(struct_pod.field(field_offset).field(), mem_ptr_ + ptr_offset);
}

}  // namespace oneflow
