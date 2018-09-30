#include "oneflow/core/register/pod_ptr.h"

namespace oneflow {

namespace {

PodPtr PodPtrField(char* ptr, const PodDesc& pod_desc, const std::string& field_name) {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(&pod_desc);
  CHECK_NOTNULL(struct_pod);
  return PodPtr(struct_pod->Field(field_name), ptr + struct_pod->PtrOffset4Field(field_name));
}

}  // namespace

bool PodPtr::HasField(const std::string& field_name) const {
  const auto* struct_pod = dynamic_cast<const StructPodDesc*>(pod_desc_);
  return struct_pod && struct_pod->HasField(field_name);
}

const PodPtr PodPtr::Field(const std::string& field_name) const {
  return PodPtrField(ptr_, *pod_desc_, field_name);
}

PodPtr PodPtr::MutField(const std::string& field_name) {
  return PodPtrField(ptr_, *pod_desc_, field_name);
}

}  // namespace oneflow
