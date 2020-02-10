#ifndef ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VPU_DESC_MSG_H_

#include <cstring>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"

namespace oneflow {

using VpuTypeId = int32_t;
using FLAT_MSG_TYPE(VpuTypeId) = VpuTypeId;
using OBJECT_MSG_TYPE(VpuTypeId) = VpuTypeId;

using VpuInstructionOpcode = int32_t;
using FLAT_MSG_TYPE(VpuInstructionOpcode) = VpuInstructionOpcode;
using OBJECT_MSG_TYPE(VpuInstructionOpcode) = VpuInstructionOpcode;

// clang-format off
BEGIN_FLAT_MSG(ConstLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
END_FLAT_MSG(ConstLogicalObjectId);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuInstructionOprand);
  FLAT_MSG_DEFINE_STRICT_ONEOF(_,
    FLAT_MSG_ONEOF_FIELD(ConstLogicalObjectId, const_oprand)
    FLAT_MSG_ONEOF_FIELD(LogicalObjectId, mutable_oprand)
    FLAT_MSG_ONEOF_FIELD(double, double_i_oprand) // i is short for immediate
    FLAT_MSG_ONEOF_FIELD(int64_t, int64_i_oprand));
END_FLAT_MSG(VpuInstructionOprand);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(AllVpuEnabledMask);
END_FLAT_MSG(AllVpuEnabledMask);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuMask);
  FLAT_MSG_DEFINE_ONEOF(mask_type,
    FLAT_MSG_ONEOF_FIELD(AllVpuEnabledMask, all_vpu_enabled)
    FLAT_MSG_ONEOF_FIELD(LogicalObjectId, enabled_parallel_desc_symbol));
END_FLAT_MSG(VpuMask);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuId);
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(VpuTypeId, vpu_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, parallel_id);

  // methods
  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
END_FLAT_MSG(VpuId);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuTypeDesc);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_machine);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_device);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, VpuTypeId, vpu_type_id);
END_OBJECT_MSG(VpuTypeDesc);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_DESC_MSG_H_
