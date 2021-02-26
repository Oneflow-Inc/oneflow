/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
  set_max_device_num_per_machine(0);
  for (const auto& pair : device_tag2device_num) {
    if (max_device_num_per_machine() < pair.second) { set_max_device_num_per_machine(pair.second); }
  }
}

void VmResourceDesc::CopyFrom(const VmResourceDesc& vm_resource_desc) {
  __Init__(vm_resource_desc.machine_num(), vm_resource_desc.device_tag2device_num());
}

int64_t VmResourceDesc::GetGlobalDeviceId(int64_t machine_id, int64_t device_id) const {
  return machine_id * max_device_num_per_machine() + device_id;
}

void VmResourceDesc::GenerateParallelConf(const char* device_tag, ParallelConf* parallel_conf) {
  const auto& device_num_iter = device_tag2device_num().find(device_tag);
  CHECK(device_num_iter != device_tag2device_num().end());
  CHECK(parallel_conf->device_name().empty());
  CHECK_GT(device_num_iter->second, 0);
  std::string device_num = std::to_string(device_num_iter->second - 1);
  parallel_conf->set_device_tag(device_tag);
  FOR_RANGE(int, i, 0, machine_num()) {
    parallel_conf->add_device_name(std::to_string(i) + ":0-" + device_num);
  }
}

}  // namespace vm
}  // namespace oneflow
