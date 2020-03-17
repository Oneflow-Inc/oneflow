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

const StreamType* LookupStreamType(StreamTypeId stream_type_id) {
  const auto& map = *StreamType4StreamTypeId();
  auto iter = map.find(stream_type_id);
  CHECK(iter != map.end());
  return iter->second;
}

void RegisterStreamType(StreamTypeId stream_type_id, const StreamType* stream_type) {
  CHECK(StreamType4StreamTypeId()->emplace(stream_type_id, stream_type).second);
}

const InstructionId& LookupInstructionId(const std::string& name) {
  const auto& map = *InstructionId4InstructionName();
  const auto& iter = map.find(name);
  CHECK(iter != map.end());
  return iter->second.Get();
}

void RegisterInstructionId(const std::string& instruction_name, StreamTypeId stream_type_id,
                           InstructionOpcode opcode, VmType type) {
  FlatMsg<InstructionId> instr_id;
  instr_id->set_stream_type_id(stream_type_id);
  instr_id->set_opcode(opcode);
  instr_id->set_type(type);
  CHECK(InstructionId4InstructionName()->emplace(instruction_name, instr_id).second);
}

}  // namespace vm
}  // namespace oneflow
