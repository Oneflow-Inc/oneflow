#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

ObjectMsgPtr<VmResourceDesc> TestUtil::NewVmResourceDesc(int64_t device_num, int64_t machine_num) {
  HashMap<std::string, int64_t> map{{"cpu", device_num}, {"gpu", device_num}};
  return ObjectMsgPtr<VmResourceDesc>::New(machine_num, map);
}

void TestUtil::AddStreamDescByInstrNames(VmDesc* vm_desc,
                                         const std::vector<std::string>& instr_names) {
  TestUtil::AddStreamDescByInstrNames(vm_desc, 1, instr_names);
}

void TestUtil::AddStreamDescByInstrNames(VmDesc* vm_desc, int64_t parallel_num,
                                         const std::vector<std::string>& instr_names) {
  auto Insert = [&](const std::string& instr_name) {
    const auto& stream_type_id = LookupInstrTypeId(instr_name).stream_type_id();
    auto instr_type = ObjectMsgPtr<StreamDesc>::New(stream_type_id, 1, parallel_num, 1);
    CHECK(vm_desc->mut_stream_type_id2desc()->Insert(instr_type.Mutable()).second);
  };
  for (const auto& instr_name : instr_names) {
    Insert(instr_name);
    Insert(std::string("Infer-") + instr_name);
  }
}

}  // namespace vm
}  // namespace oneflow
