#include "oneflow/core/register/pod_ptr.h"
#include "oneflow/core/register/pod_helper.h"

namespace oneflow {

bool PodPtr::HasField(const std::string& field_name) const {
  return PodHelper(*pod_proto_).HasField(field_name);
}

PodPtr PodPtr::Field(const std::string& field_name) const {
  PodHelper pod_helper(*pod_proto_);
  return PodPtr(pod_helper.Field(field_name).pod_proto(),
                mem_ptr_ + pod_helper.PtrOffset4Field(field_name));
}

}  // namespace oneflow
