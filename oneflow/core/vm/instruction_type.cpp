#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

HashMap<std::string, InstrTypeId>* InstrTypeId4InstructionName() {
  static HashMap<std::string, InstrTypeId> map;
  return &map;
}

std::map<InstrTypeId, const InstructionType*>* InstructionType4InstrTypeId() {
  static std::map<InstrTypeId, const InstructionType*> map;
  return &map;
}

}  // namespace

const InstrTypeId& LookupInstrTypeId(const std::string& name) {
  const auto& map = *InstrTypeId4InstructionName();
  const auto& iter = map.find(name);
  CHECK(iter != map.end());
  return iter->second;
}

void ForEachInstrTypeId(std::function<void(const InstrTypeId&)> DoEach) {
  for (const auto& pair : *InstrTypeId4InstructionName()) { DoEach(pair.second); }
}

void RegisterInstructionType(const std::string& instruction_name,
                             const std::type_index& stream_type_index,
                             const std::type_index& instr_type_index, InterpretType interpret_type,
                             VmType vm_type, const InstructionType* instruction_type) {
  InstrTypeId instr_type_id;
  instr_type_id.__Init__(stream_type_index, instr_type_index, interpret_type, vm_type);
  CHECK(InstrTypeId4InstructionName()->emplace(instruction_name, instr_type_id).second);
  auto ret = InstructionType4InstrTypeId()->emplace(instr_type_id, instruction_type);
  if (instruction_type == nullptr) { return; }
  if (!ret.second) { CHECK(typeid(ret.first->second) == typeid(instruction_type)); }
}

const InstructionType* LookupInstructionType(const InstrTypeId& instr_type_id) {
  return InstructionType4InstrTypeId()->at(instr_type_id);
}

}  // namespace vm
}  // namespace oneflow
