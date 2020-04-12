#ifndef ONEFLOW_CORE_VM_CONST_OBJECT_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_CONST_OBJECT_INSTRUCTION_TYPE_H_

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
FLAT_MSG_VIEW_BEGIN(ConstObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(InitConstHostOperand, serialized_logical_object_id);
FLAT_MSG_VIEW_END(ConstObjectInstrOperand);
// clang-format on

template<typename T, typename SerializedT>
class InitConstObjectInstructionType final : public InstructionType {
 public:
  InitConstObjectInstructionType() = default;
  ~InitConstObjectInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(InstrCtx* instr_ctx) const override {
    FlatMsgView<ConstObjectInstrOperand> args(instr_ctx->instr_msg().operand());
    FOR_RANGE(int, i, 0, args->serialized_logical_object_id_size()) {
      const auto& operand = args->serialized_logical_object_id(i);
      int64_t logical_object_id = operand.logical_object_id();
      const auto& serialized_conf = Global<Storage<SerializedT>>::Get()->Get(logical_object_id);
      auto* mirrored_object = instr_ctx->mut_operand_type(operand);
      mirrored_object->Init<ObjectWrapper<T>>(serialized_conf);
    }
  }
  void Compute(InstrCtx* instr_ctx) const override {
    // do nothing
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONST_OBJECT_INSTRUCTION_TYPE_H_
