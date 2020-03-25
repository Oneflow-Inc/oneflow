#ifndef ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_

#include <glog/logging.h>

namespace oneflow {
namespace vm {

class InstructionMsg;
class Instruction;
class Scheduler;

class InstructionType {
 public:
  virtual ~InstructionType() = default;

  virtual void Compute(Instruction* instruction) const = 0;
  virtual void Infer(Instruction* instruction) const { LOG(FATAL) << "UNIMPLEMENTED"; }

  virtual void Compute(Scheduler* scheduler, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }

 protected:
  InstructionType() = default;
};

class InstrTypeId;
const InstrTypeId& LookupInstrTypeId(const std::string& instr_type_name);
void ForEachInstrTypeId(std::function<void(const InstrTypeId&)> DoEach);
void RegisterInstrTypeId(const std::string& instr_type_name,
                         const std::type_index& stream_type_index, InstructionOpcode opcode,
                         VmType type, const InstructionType* instruction_type);
template<typename T>
void RegisterInstrTypeId(const std::string& instr_type_name, InstructionOpcode opcode,
                         VmType type) {
  RegisterInstrTypeId(instr_type_name, typeid(T), opcode, type, nullptr);
}

template<typename T>
void RegisterInstrTypeId(const std::string& instr_type_name, VmType type) {
  RegisterInstrTypeId(instr_type_name, typeid(typename T::stream_type), T::opcode, type, new T());
}

const InstructionType* LookupInstructionType(const InstrTypeId&);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
