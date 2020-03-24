#ifndef ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_

#include <glog/logging.h>

namespace oneflow {
namespace vm {

class InstrChain;
class InstrChain;
class Scheduler;

class InstructionType {
 public:
  virtual ~InstructionType() = default;

  virtual void Compute(InstrChain* instr_chain) const = 0;
  virtual void Infer(InstrChain* instr_chain) const { LOG(FATAL) << "UNIMPLEMENTED"; }

  virtual void Compute(Scheduler* scheduler, InstrChain* instr_chain) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Infer(Scheduler* scheduler, InstrChain* instr_chain) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
