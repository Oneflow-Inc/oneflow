#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

PodPtr PodPtrField(const PodDesc* pod_desc, const FieldId& field_id, char* ptr) {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(pod_desc);
  CHECK_NOTNULL(struct_pod);
  return PodPtr(struct_pod->Field(field_id), ptr + struct_pod->ByteOffset4Field(field_id));
}

bool PodPtr::HasField(const FieldId& field_id) const {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(pod_desc_);
  return struct_pod && struct_pod->HasField(field_id);
}

const PodPtr PodPtr::Field(const FieldId& field_id) const {
  return PodPtrField(pod_desc_, field_id, ptr_);
}

PodPtr PodPtr::MutField(const FieldId& field_id) { return PodPtrField(pod_desc_, field_id, ptr_); }

}  // namespace oneflow
