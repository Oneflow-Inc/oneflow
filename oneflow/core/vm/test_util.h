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

class TestResourceDescScope final {
 public:
  TestResourceDescScope(const TestResourceDescScope&) = delete;
  TestResourceDescScope(TestResourceDescScope&&) = delete;
  TestResourceDescScope(int64_t gpu_device_num, int64_t cpu_device_num);
  ~TestResourceDescScope();
};

struct TestUtil {
  using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc() { return NewVmResourceDesc(1); }

  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num) {
    return NewVmResourceDesc(device_num, 1);
  }

  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num, int64_t machine_num);

  // return logical_object_id
  static int64_t NewObject(InstructionMsgList* list, const std::string& device_name) {
    int64_t parallel_desc_symbol_id = 0;
    return NewObject(list, device_name, &parallel_desc_symbol_id);
  }
  static int64_t NewObject(InstructionMsgList*, const std::string& device_name,
                           int64_t* parallel_desc_symbol_id);
  static int64_t NewSymbol(InstructionMsgList*);
  static int64_t NewStringSymbol(InstructionMsgList*, const std::string& str);

  static void AddStreamDescByInstrNames(VmDesc* vm_desc,
                                        const std::vector<std::string>& instr_names);
  static void AddStreamDescByInstrNames(VmDesc* vm_desc, int64_t parallel_num,
                                        const std::vector<std::string>& instr_names);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TEST_UTIL_H_
