#ifndef ONEFLOW_CORE_VM_TEST_UTIL_H_
#define ONEFLOW_CORE_VM_TEST_UTIL_H_

#include <vector>
#include <string>
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/vm_resource_desc.msg.h"

namespace oneflow {
namespace vm {

class VmDesc;
class VmResourceDesc;

struct TestUtil {
  using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc() { return NewVmResourceDesc(1); }
  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num) {
    return NewVmResourceDesc(device_num, 1);
  }
  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num, int64_t machine_num);

  // return logical_object_id
  static int64_t NewObject(InstructionMsgList*, const std::string& device_name);

  static void AddStreamDescByInstrNames(VmDesc* vm_desc,
                                        const std::vector<std::string>& instr_names);
  static void AddStreamDescByInstrNames(VmDesc* vm_desc, int64_t parallel_num,
                                        const std::vector<std::string>& instr_names);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TEST_UTIL_H_
