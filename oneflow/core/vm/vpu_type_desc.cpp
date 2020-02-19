#include "oneflow/core/vm/vpu_type_desc.msg.h"

namespace oneflow {

int32_t VpuTypeDesc::num_threads() const {
  int32_t num_devices = num_machines() * num_devices_per_machine();
  CHECK_EQ(num_devices % num_streams(), 0);
  return num_devices / num_streams();
}

}  // namespace oneflow
