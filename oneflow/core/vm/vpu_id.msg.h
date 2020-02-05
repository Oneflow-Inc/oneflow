#ifndef ONEFLOW_CORE_VM_VPU_ID_MSG_H_
#define ONEFLOW_CORE_VM_VPU_ID_MSG_H_

#include <cstring>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/vpu_instruction_desc.msg.h"

namespace oneflow {

// clang-format off
BEGIN_FLAT_MSG(GlobalDeviceId);
  FLAT_MSG_DEFINE_OPTIONAL(int32_t, machine_id);
  FLAT_MSG_DEFINE_OPTIONAL(int32_t, device_id);
END_FLAT_MSG(GlobalDeviceId);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuId);
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(VpuInstructionDesc::_FLAT_MSG_ONEOF_ENUM_TYPE(vpu_type), vpu_type);
  FLAT_MSG_DEFINE_OPTIONAL(GlobalDeviceId, global_device_id);

  // methods
  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
END_FLAT_MSG(VpuId);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_ID_MSG_H_
