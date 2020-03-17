#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

HashMap<StreamTypeId, const StreamType*>* StreamType4StreamTypeId() {
  static HashMap<StreamTypeId, const StreamType*> map;
  return &map;
}

HashMap<std::string, FlatMsg<InstructionId>>* InstructionId4InstructionName() {
  static HashMap<std::string, FlatMsg<InstructionId>> map;
  return &map;
}

}  // namespace

const StreamType* LookupStreamType(StreamTypeId vm_stream_type_id) {
  const auto& map = *StreamType4StreamTypeId();
  auto iter = map.find(vm_stream_type_id);
  CHECK(iter != map.end());
  return iter->second;
}

void RegisterStreamType(StreamTypeId vm_stream_type_id, const StreamType* vm_stream_type) {
  CHECK(StreamType4StreamTypeId()->emplace(vm_stream_type_id, vm_stream_type).second);
}

const InstructionId& LookupInstructionId(const std::string& name) {
  const auto& map = *InstructionId4InstructionName();
  const auto& iter = map.find(name);
  CHECK(iter != map.end());
  return iter->second.Get();
}

void RegisterInstructionId(const std::string& vm_instruction_name, StreamTypeId vm_stream_type_id,
                           InstructionOpcode opcode, VmType vm_type) {
  FlatMsg<InstructionId> vm_instr_id;
  vm_instr_id->set_vm_stream_type_id(vm_stream_type_id);
  vm_instr_id->set_opcode(opcode);
  vm_instr_id->set_vm_type(vm_type);
  CHECK(InstructionId4InstructionName()->emplace(vm_instruction_name, vm_instr_id).second);
}

}  // namespace vm
}  // namespace oneflow
