#ifndef ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_

#include <glog/logging.h>

namespace oneflow {
namespace vm {

class InstrChain;
class InstrChain;
class Instruction;
class Scheduler;

class InstructionType {
 public:
  virtual ~InstructionType() = default;

  virtual void Compute(Instruction* instruction) const = 0;
  virtual void Infer(Instruction* instruction) const { LOG(FATAL) << "UNIMPLEMENTED"; }

  virtual void Compute(Scheduler* scheduler, InstrChain* instr_chain) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Infer(Scheduler* scheduler, InstrChain* instr_chain) const {
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
                         VmType type);
template<typename T>
void RegisterInstrTypeId(const std::string& instr_type_name, InstructionOpcode opcode,
                         VmType type) {
  RegisterInstrTypeId(instr_type_name, typeid(T), opcode, type);
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
