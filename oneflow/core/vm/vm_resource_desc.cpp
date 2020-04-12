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

int64_t VmResourceDesc::GetGlobalDeviceId(const ParallelDesc& parallel_desc,
                                          int64_t parallel_id) const {
  int64_t machine_parallel_id = parallel_id / parallel_desc.device_num_of_each_machine();
  int64_t device_parallel_id = parallel_id % parallel_desc.device_num_of_each_machine();
  int64_t machine_id = parallel_desc.sorted_machine_ids().at(machine_parallel_id);
  int64_t device_id = parallel_desc.sorted_dev_phy_ids(machine_id).at(device_parallel_id);
  int64_t device_num =
      device_tag2device_num().at(DeviceTag4DeviceType(parallel_desc.device_type()));
  return machine_id * device_num + device_id;
}

}  // namespace vm
}  // namespace oneflow
