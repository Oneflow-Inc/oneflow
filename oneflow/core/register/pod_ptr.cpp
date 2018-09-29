#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

bool PodPtr::HasField(const std::string& field_name) const {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(pod_desc_);
  return struct_pod && struct_pod->HasField(field_name);
}

PodPtr PodPtr::Field(const std::string& field_name) const {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(pod_desc_);
  CHECK_NOTNULL(struct_pod);
  return PodPtr(struct_pod->Field(field_name), ptr_ + struct_pod->PtrOffset4Field(field_name));
}

}  // namespace oneflow
