#include "oneflow/core/vm/vm_stream_desc.msg.h"

namespace oneflow {

int32_t VmStreamDesc::num_threads() const {
  int32_t num_devices = num_machines() * num_devices_per_machine();
  CHECK_EQ(num_devices % num_streams_per_thread(), 0);
  return num_devices / num_streams_per_thread();
}

}  // namespace oneflow
