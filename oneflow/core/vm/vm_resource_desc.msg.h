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
#ifndef ONEFLOW_CORE_VM_VM_RESOURCE_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VM_RESOURCE_DESC_MSG_H_

#include <unordered_map>
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class ParallelConf;

namespace vm {

using DeviceTag2DeviceNum = std::unordered_map<std::string, int64_t>;

// clang-format off
OBJECT_MSG_BEGIN(VmResourceDesc);
  // methods
  OF_PUBLIC void __Init__(const Resource& resource);
  OF_PUBLIC void __Init__(
      int64_t machine_num, const DeviceTag2DeviceNum& device_tag2device_num);
  OF_PUBLIC void CopyFrom(const VmResourceDesc& vm_resource_desc);
  OF_PUBLIC int64_t GetGlobalDeviceId(int64_t machine_id, int64_t device_id) const;
  OF_PUBLIC void GenerateParallelConf(const char* device_tag, ParallelConf* parallel_conf);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, machine_num);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, max_device_num_per_machine);
  OBJECT_MSG_DEFINE_STRUCT(DeviceTag2DeviceNum, device_tag2device_num);
OBJECT_MSG_END(VmResourceDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_RESOURCE_DESC_MSG_H_
