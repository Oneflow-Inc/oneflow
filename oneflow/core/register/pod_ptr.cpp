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
