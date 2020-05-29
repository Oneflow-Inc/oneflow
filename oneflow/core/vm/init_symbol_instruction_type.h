#ifndef ONEFLOW_CORE_VM_INIT_SYMBOL_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_INIT_SYMBOL_INSTRUCTION_TYPE_H_

#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/object_wrapper.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_VIEW_BEGIN(SymbolInstrOperand);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(InitSymbolOperand, serialized_logical_object_id);
FLAT_MSG_VIEW_END(SymbolInstrOperand);
// clang-format on

template<typename T, typename SerializedT>
class InitSymbolInstructionType final : public InstructionType {
 public:
  InitSymbolInstructionType() = default;
  ~InitSymbolInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(Instruction* instruction) const override {
    FlatMsgView<SymbolInstrOperand> args(instruction->instr_msg().operand());
    FOR_RANGE(int, i, 0, args->serialized_logical_object_id_size()) {
      const auto& operand = args->serialized_logical_object_id(i);
      int64_t logical_object_id = operand.logical_object_id();
      const auto& serialized_conf = Global<Storage<SerializedT>>::Get()->Get(logical_object_id);
      auto* rw_mutexed_object = instruction->mut_operand_type(operand);
      rw_mutexed_object->Init<ObjectWrapper<T>>(serialized_conf);
    }
  }
  void Compute(Instruction* instruction) const override {
    // do nothing
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INIT_SYMBOL_INSTRUCTION_TYPE_H_
