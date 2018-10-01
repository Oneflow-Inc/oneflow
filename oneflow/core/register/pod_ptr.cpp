#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

bool PodPtr::HasField(const FieldId& field_id) const {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(pod_desc_);
  return struct_pod && struct_pod->HasField(field_id);
}

PodPtr PodPtr::Field(const FieldId& field_id) const {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(pod_desc_);
  CHECK_NOTNULL(struct_pod);
  return PodPtr(struct_pod->Field(field_id), ptr_ + struct_pod->ByteOffset4Field(field_id));
}

}  // namespace oneflow
