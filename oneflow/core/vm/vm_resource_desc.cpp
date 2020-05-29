#include "oneflow/core/vm/vm_resource_desc.msg.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/common/util.h"

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

int64_t VmResourceDesc::GetGlobalDeviceId(int64_t machine_id, const std::string& device_tag,
                                          int64_t device_id) const {
  int64_t device_num = device_tag2device_num().at(device_tag);
  return machine_id * device_num + device_id;
}

void VmResourceDesc::GenerateParallelConf(const char* device_tag, ParallelConf* parallel_conf) {
  const auto& device_num_iter = device_tag2device_num().find(device_tag);
  CHECK(device_num_iter != device_tag2device_num().end());
  CHECK(parallel_conf->device_name().empty());
  CHECK_GT(device_num_iter->second, 0);
  std::string device_num = std::to_string(device_num_iter->second - 1);
  FOR_RANGE(int, i, 0, machine_num()) {
    parallel_conf->add_device_name(std::to_string(i) + ":" + device_tag + ":0-" + device_num);
  }
}

}  // namespace vm
}  // namespace oneflow
