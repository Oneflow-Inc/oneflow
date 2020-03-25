#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

HashMap<std::string, FlatMsg<InstrTypeId>>* InstrTypeId4InstructionName() {
  static HashMap<std::string, FlatMsg<InstrTypeId>> map;
  return &map;
}

HashMap<const InstrTypeId*, const InstructionType*>* InstructionType4InstrTypeId() {
  static HashMap<const InstrTypeId*, const InstructionType*> map;
  return &map;
}

}  // namespace

const InstrTypeId& LookupInstrTypeId(const std::string& name) {
  const auto& map = *InstrTypeId4InstructionName();
  const auto& iter = map.find(name);
  CHECK(iter != map.end());
  return iter->second.Get();
}

void ForEachInstrTypeId(std::function<void(const InstrTypeId&)> DoEach) {
  for (const auto& pair : *InstrTypeId4InstructionName()) { DoEach(pair.second.Get()); }
}

void RegisterInstrTypeId(const std::string& instruction_name,
                         const std::type_index& stream_type_index, InstructionOpcode opcode,
                         VmType type) {
  auto Register = [&](const std::string& instruction_name, InterpretType interpret_type) {
    FlatMsg<InstrTypeId> instr_type_id;
    instr_type_id->__Init__(stream_type_index, interpret_type, opcode, type);
    CHECK(InstrTypeId4InstructionName()->emplace(instruction_name, instr_type_id).second);
  };
  Register(instruction_name, InterpretType::kCompute);
  Register(std::string("Infer-") + instruction_name, InterpretType::kInfer);
}

}  // namespace vm
}  // namespace oneflow
