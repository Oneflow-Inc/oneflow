#ifndef ONEFLOW_CORE_VM_TEST_UTIL_H_
#define ONEFLOW_CORE_VM_TEST_UTIL_H_

#include <vector>
#include <string>

namespace oneflow {
namespace vm {

class VmDesc;

struct TestUtil {
  static void AddStreamDescByInstrNames(VmDesc* vm_desc,
                                        const std::vector<std::string>& instr_names);
  static void AddStreamDescByInstrNames(VmDesc* vm_desc, int64_t parallel_num,
                                        const std::vector<std::string>& instr_names);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TEST_UTIL_H_
