#ifndef ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_

#include <glog/logging.h>
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/infer_stream_type.h"

namespace oneflow {
namespace vm {

class InstructionMsg;
class InstrCtx;
class Scheduler;

class InstructionType {
 public:
  virtual ~InstructionType() = default;

  virtual void Compute(InstrCtx* instr_ctx) const = 0;
  virtual void Infer(InstrCtx* instr_ctx) const { LOG(FATAL) << "UNIMPLEMENTED"; }

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
void RegisterInstructionType(const std::string& instr_type_name,
                             const std::type_index& stream_type_index,
                             const std::type_index& instr_type_index, InterpretType interpret_type,
                             VmType vm_type, const InstructionType* instruction_type);

template<typename T>
void RegisterInstructionType(const std::string& instr_type_name) {
  TryRegisterStreamType(typeid(typename T::stream_type), InterpretType::kCompute,
                        new typename T::stream_type());
  RegisterInstructionType(instr_type_name, typeid(typename T::stream_type), typeid(T),
                          InterpretType::kCompute, VmType::kRemote, new T());
  TryRegisterStreamType(typeid(InferStreamType<typename T::stream_type>), InterpretType::kInfer,
                        new InferStreamType<typename T::stream_type>());
  RegisterInstructionType(std::string("Infer-") + instr_type_name,
                          typeid(InferStreamType<typename T::stream_type>), typeid(T),
                          InterpretType::kInfer, VmType::kRemote, new T());
  RegisterInstructionType(std::string("LocalInfer-") + instr_type_name,
                          typeid(InferStreamType<typename T::stream_type>), typeid(T),
                          InterpretType::kInfer, VmType::kLocal, new T());
}

template<typename T>
void RegisterLocalInstructionType(const std::string& instr_type_name) {
  TryRegisterStreamType(typeid(typename T::stream_type), InterpretType::kCompute,
                        new typename T::stream_type());
  RegisterInstructionType(instr_type_name, typeid(typename T::stream_type), typeid(T),
                          InterpretType::kCompute, VmType::kLocal, new T());
  TryRegisterStreamType(typeid(InferStreamType<typename T::stream_type>), InterpretType::kInfer,
                        new InferStreamType<typename T::stream_type>());
  RegisterInstructionType(std::string("Infer-") + instr_type_name,
                          typeid(InferStreamType<typename T::stream_type>), typeid(T),
                          InterpretType::kInfer, VmType::kLocal, new T());
}

const InstructionType* LookupInstructionType(const InstrTypeId&);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
