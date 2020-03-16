#include "oneflow/core/vm/vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

HashMap<VmStreamTypeId, const VmStreamType*>* VmStreamType4VmStreamTypeId() {
  static HashMap<VmStreamTypeId, const VmStreamType*> map;
  return &map;
}

HashMap<std::string, FlatMsg<VmInstructionId>>* VmInstructionId4VmInstructionName() {
  static HashMap<std::string, FlatMsg<VmInstructionId>> map;
  return &map;
}

}  // namespace

const VmStreamType* LookupVmStreamType(VmStreamTypeId vm_stream_type_id) {
  const auto& map = *VmStreamType4VmStreamTypeId();
  auto iter = map.find(vm_stream_type_id);
  CHECK(iter != map.end());
  return iter->second;
}

void RegisterVmStreamType(VmStreamTypeId vm_stream_type_id, const VmStreamType* vm_stream_type) {
  CHECK(VmStreamType4VmStreamTypeId()->emplace(vm_stream_type_id, vm_stream_type).second);
}

const VmInstructionId& LookupVmInstructionId(const std::string& name) {
  const auto& map = *VmInstructionId4VmInstructionName();
  const auto& iter = map.find(name);
  CHECK(iter != map.end());
  return iter->second.Get();
}

void RegisterVmInstructionId(const std::string& vm_instruction_name,
                             VmStreamTypeId vm_stream_type_id, VmInstructionOpcode opcode,
                             VmType vm_type) {
  FlatMsg<VmInstructionId> vm_instr_id;
  vm_instr_id->set_vm_stream_type_id(vm_stream_type_id);
  vm_instr_id->set_opcode(opcode);
  vm_instr_id->set_vm_type(vm_type);
  CHECK(VmInstructionId4VmInstructionName()->emplace(vm_instruction_name, vm_instr_id).second);
}

}  // namespace oneflow
