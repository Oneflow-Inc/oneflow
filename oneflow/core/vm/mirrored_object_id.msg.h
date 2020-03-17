#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/instruction.pb.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_BEGIN(MirroredParallelId);
FLAT_MSG_END(MirroredParallelId);

FLAT_MSG_BEGIN(AllParallelId);
FLAT_MSG_END(AllParallelId);

FLAT_MSG_BEGIN(MirroredObjectOperand);
  // methods
  PUBLIC void __Init__(const LogicalObjectId& logical_object_id, int64_t parallel_id);
  PUBLIC void __Init__(const LogicalObjectId& logical_object_id);
  PUBLIC void __Init__(const MirroredObjectOperandProto& proto);
  PUBLIC int64_t GetParallelId(int64_t default_parallel_id) const;

  // fields
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, logical_object_id);
  FLAT_MSG_DEFINE_ONEOF(operand_type,
    FLAT_MSG_ONEOF_FIELD(int64_t, fixed_parallel_id)
    FLAT_MSG_ONEOF_FIELD(MirroredParallelId, mirrored_parallel_id)
    FLAT_MSG_ONEOF_FIELD(AllParallelId, all_parallel_id));
FLAT_MSG_END(MirroredObjectOperand);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(MirroredObjectId);
  // methods
  PUBLIC void __Init__(uint64_t logical_object_id_value, int64_t parallel_id);
  PUBLIC void __Init__(const MirroredObjectOperand& operand, int64_t parallel_id);
  PUBLIC FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();

  // fields
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, logical_object_id_value);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, parallel_id);

FLAT_MSG_END(MirroredObjectId);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_
