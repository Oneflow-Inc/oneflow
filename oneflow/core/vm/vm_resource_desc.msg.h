#ifndef ONEFLOW_CORE_VM_VM_RESOURCE_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VM_RESOURCE_DESC_MSG_H_

#include <unordered_map>
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class ParallelConf;

namespace vm {

using DeviceTag2DeviceNum = std::unordered_map<std::string, int64_t>;

// clang-format off
OBJECT_MSG_BEGIN(VmResourceDesc);
  // methods
  PUBLIC void __Init__(const Resource& resource);
  PUBLIC void __Init__(
      int64_t machine_num, const DeviceTag2DeviceNum& device_tag2device_num);
  PUBLIC void CopyFrom(const VmResourceDesc& vm_resource_desc);
  PUBLIC int64_t GetGlobalDeviceId(int64_t machine_id, const std::string& device_tag, int64_t device_id) const;
  PUBLIC void GenerateParallelConf(const char* device_tag, ParallelConf* parallel_conf);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, machine_num);
  OBJECT_MSG_DEFINE_STRUCT(DeviceTag2DeviceNum, device_tag2device_num);
OBJECT_MSG_END(VmResourceDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_RESOURCE_DESC_MSG_H_
