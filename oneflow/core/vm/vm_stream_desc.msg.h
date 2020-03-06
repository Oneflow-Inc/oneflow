#ifndef ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VPU_DESC_MSG_H_

#include <cstring>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vm_instruction_operand.msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"

namespace oneflow {

using VmStreamTypeId = int;
using VmInstructionOpcode = int32_t;

// clang-format off
FLAT_MSG_BEGIN(AllVmStreamEnabledMask);
FLAT_MSG_END(AllVmStreamEnabledMask);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(VmStreamMask);
  FLAT_MSG_DEFINE_ONEOF(mask_type,
    FLAT_MSG_ONEOF_FIELD(AllVmStreamEnabledMask, all_vm_stream_enabled)
    FLAT_MSG_ONEOF_FIELD(LogicalObjectId, enabled_parallel_desc_symbol));
FLAT_MSG_END(VmStreamMask);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(VmStreamId);
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(VmStreamTypeId, vm_stream_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, parallel_id);

  // methods
  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
FLAT_MSG_END(VmStreamId);
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(VmStreamDesc);
  // methods
  PUBLIC void __Init__(VmStreamTypeId vm_stream_type_id, int32_t num_machines, int32_t num_streams_per_machine,
                       int32_t num_streams_per_thread);
  PUBLIC int32_t num_threads() const;
  PUBLIC int32_t parallel_num() const { return num_machines() * num_streams_per_machine(); }

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_machines);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_streams_per_machine);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_streams_per_thread);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, VmStreamTypeId, vm_stream_type_id);
OBJECT_MSG_END(VmStreamDesc);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
