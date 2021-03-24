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
#include "oneflow/core/vm/mirrored_object_id.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

template<typename T>
void InitFromProto(Operand* that, const T& proto) {
  that->set_logical_object_id(proto.logical_object_id());
  if (proto.has_sole_mirrored_object()) {
    that->mutable_sole_mirrored_object();
  } else if (proto.has_current_global_device_id()) {
    that->mutable_current_global_device_id();
  } else if (proto.has_all_mirrored_object()) {
    that->mutable_all_mirrored_object();
  } else {
    UNIMPLEMENTED();
  }
}
}

void Operand::__Init__(const ObjectId& logical_object_id) {
  set_logical_object_id(logical_object_id);
  mutable_current_global_device_id();
}

void Operand::__Init__(const ObjectId& logical_object_id, const SoleMirroredObject&) {
  set_logical_object_id(logical_object_id);
  mutable_sole_mirrored_object();
}

void Operand::__Init__(const ObjectId& logical_object_id, const AllMirroredObject&) {
  set_logical_object_id(logical_object_id);
  mutable_all_mirrored_object();
}

void Operand::__Init__(const OperandProto& proto) { InitFromProto(this, proto); }
void Operand::__Init__(const cfg::OperandProto& proto) { InitFromProto(this, proto); }

void Operand::ToProto(OperandProto* proto) const {
  proto->set_logical_object_id(logical_object_id());
  if (has_sole_mirrored_object()) {
    proto->mutable_sole_mirrored_object();
  } else if (has_current_global_device_id()) {
    proto->mutable_current_global_device_id();
  } else if (has_all_mirrored_object()) {
    proto->mutable_all_mirrored_object();
  } else {
    UNIMPLEMENTED();
  }
}

int64_t Operand::GetGlobalDeviceId(int64_t current_global_device_id) const {
  if (has_sole_mirrored_object()) { return 0; }
  CHECK(has_current_global_device_id());
  return current_global_device_id;
}

void MirroredObjectId::__Init__(int64_t logical_object_id_value, int64_t global_device_id) {
  set_logical_object_id_value(logical_object_id_value);
  set_global_device_id(global_device_id);
}

}  // namespace vm
}  // namespace oneflow
