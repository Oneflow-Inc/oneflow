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
#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID__H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID__H_

#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/instruction.cfg.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_BEGIN(SoleMirroredObject);
FLAT_MSG_END(SoleMirroredObject);

FLAT_MSG_BEGIN(CurrentGlobalDeviceId);
FLAT_MSG_END(CurrentGlobalDeviceId);

FLAT_MSG_BEGIN(AllMirroredObject);
FLAT_MSG_END(AllMirroredObject);

FLAT_MSG_BEGIN(Operand);
 public:
  // methods
  // init current_global_device_id
  void __Init__(const ObjectId& logical_object_id);
  // init sole_mirrored_object
  void __Init__(const ObjectId& logical_object_id, const SoleMirroredObject&);
  // init all_mirrored_object
  void __Init__(const ObjectId& logical_object_id, const AllMirroredObject&);
  void __Init__(const OperandProto& proto);
  void __Init__(const cfg::OperandProto& proto);
  void ToProto(OperandProto* proto) const;
  int64_t GetGlobalDeviceId(int64_t default_global_device_id) const;

 private:
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(ObjectId, logical_object_id);
  FLAT_MSG_DEFINE_ONEOF(operand_type,
    FLAT_MSG_ONEOF_FIELD(CurrentGlobalDeviceId, current_global_device_id)
    FLAT_MSG_ONEOF_FIELD(SoleMirroredObject, sole_mirrored_object)
    FLAT_MSG_ONEOF_FIELD(AllMirroredObject, all_mirrored_object));
FLAT_MSG_END(Operand);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(MirroredObjectId);
 public:
  // methods
  void __Init__() {}
  void __Init__(int64_t logical_object_id_value, int64_t global_device_id);
  template<int64_t(*TransformLogicalObjectId)(int64_t)>
         void __Init__(const Operand& operand, int64_t global_device_id) {
    __Init__(TransformLogicalObjectId(operand.logical_object_id()),
             operand.GetGlobalDeviceId(global_device_id));
  }
  void __Init__(const Operand& operand, int64_t global_device_id) {
    __Init__(operand.logical_object_id(), operand.GetGlobalDeviceId(global_device_id));
  }
  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();

 private:
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, logical_object_id_value);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, global_device_id);
FLAT_MSG_END(MirroredObjectId);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID__H_
