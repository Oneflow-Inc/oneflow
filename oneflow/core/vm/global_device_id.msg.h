#ifndef ONEFLOW_CORE_VM_GLOBAL_DEVICE_ID_MSG_H_
#define ONEFLOW_CORE_VM_GLOBAL_DEVICE_ID_MSG_H_

#include <cstring>
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

// clang-format off
BEGIN_FLAT_MSG(GlobalDeviceId);
  FLAT_MSG_DEFINE_OPTIONAL(int32_t, machine_id);
  FLAT_MSG_DEFINE_OPTIONAL(int32_t, device_id);
END_FLAT_MSG(GlobalDeviceId);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_GLOBAL_DEVICE_ID_MSG_H_
