#include "oneflow/core/vm/vm_resource_desc.msg.h"

namespace oneflow {
namespace vm {

void VmResourceDesc::__Init__(const Resource& resource) {
  __Init__(resource.machine_num(),
           {{"cpu", resource.cpu_device_num()}, {"gpu", resource.gpu_device_num()}});
}

void VmResourceDesc::__Init__(int64_t machine_num,
                              const DeviceTag2DeviceNum& device_tag2device_num) {
  set_machine_num(machine_num);
  *mutable_device_tag2device_num() = device_tag2device_num;
}

void VmResourceDesc::CopyFrom(const VmResourceDesc& vm_resource_desc) {
  __Init__(vm_resource_desc.machine_num(), vm_resource_desc.device_tag2device_num());
}

}  // namespace vm
}  // namespace oneflow
