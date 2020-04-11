#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/logical_object_id.h"
#include "oneflow/core/vm/instruction.pb.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_BEGIN(CurrentGlobalDeviceId);
FLAT_MSG_END(CurrentGlobalDeviceId);

FLAT_MSG_BEGIN(AllMirrored);
FLAT_MSG_END(AllMirrored);

FLAT_MSG_BEGIN(Operand);
  // methods
  // init current_global_device_id
  PUBLIC void __Init__(const LogicalObjectId& logical_object_id);
  // init fixed_global_device_id
  PUBLIC void __Init__(const LogicalObjectId& logical_object_id, int64_t global_device_id);
  // init all_mirrored
  PUBLIC void __Init__(const LogicalObjectId& logical_object_id, const AllMirrored&);
  PUBLIC void __Init__(const OperandProto& proto);
  PUBLIC int64_t GetGlobalDeviceId(int64_t default_global_device_id) const;

  // fields
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, logical_object_id);
  FLAT_MSG_DEFINE_ONEOF(operand_type,
    FLAT_MSG_ONEOF_FIELD(int64_t, fixed_global_device_id)
    FLAT_MSG_ONEOF_FIELD(CurrentGlobalDeviceId, current_global_device_id)
    FLAT_MSG_ONEOF_FIELD(AllMirrored, all_mirrored));
FLAT_MSG_END(Operand);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(MirroredObjectId);
  // methods
  PUBLIC void __Init__() {}
  PUBLIC void __Init__(int64_t logical_object_id_value, int64_t global_device_id);
  PUBLIC template<int64_t(*TransformLogicalObjectId)(int64_t)>
         void __Init__(const Operand& operand, int64_t global_device_id) {
    __Init__(TransformLogicalObjectId(operand.logical_object_id()),
             operand.GetGlobalDeviceId(global_device_id));
  }
  PUBLIC void __Init__(const Operand& operand, int64_t global_device_id) {
    __Init__(operand.logical_object_id(), operand.GetGlobalDeviceId(global_device_id));
  }
  PUBLIC FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();

  // fields
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, logical_object_id_value);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, global_device_id);

FLAT_MSG_END(MirroredObjectId);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_
