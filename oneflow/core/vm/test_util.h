#ifndef ONEFLOW_CORE_VM_TEST_UTIL_H_
#define ONEFLOW_CORE_VM_TEST_UTIL_H_

#include <vector>
#include <string>
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vm_resource_desc.msg.h"

namespace oneflow {
namespace vm {

class VmDesc;
class VmResourceDesc;

struct TestUtil {
  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc() { return NewVmResourceDesc(1); }
  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num) {
    return NewVmResourceDesc(device_num, 1);
  }
  static ObjectMsgPtr<VmResourceDesc> NewVmResourceDesc(int64_t device_num, int64_t machine_num);

  static void AddStreamDescByInstrNames(VmDesc* vm_desc,
                                        const std::vector<std::string>& instr_names);
  static void AddStreamDescByInstrNames(VmDesc* vm_desc, int64_t parallel_num,
                                        const std::vector<std::string>& instr_names);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TEST_UTIL_H_
