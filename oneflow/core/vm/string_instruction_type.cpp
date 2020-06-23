#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/storage.h"

namespace oneflow {
namespace vm {

COMMAND(Global<Storage<std::string>>::SetAllocated(new Storage<std::string>()));

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(StringObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::InitSymbolOperand, string);
FLAT_MSG_VIEW_END(StringObjectInstrOperand);
// clang-format on

}  // namespace

class InitStringSymbolInstructionType final : public InstructionType {
 public:
  InitStringSymbolInstructionType() = default;
  ~InitStringSymbolInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(Instruction* instruction) const override {
    FlatMsgView<StringObjectInstrOperand> args(instruction->instr_msg().operand());
    FOR_RANGE(int, i, 0, args->string_size()) {
      int64_t logical_object_id = args->string(i).logical_object_id();
      const auto& str = Global<Storage<std::string>>::Get()->Get(logical_object_id);
      auto* rw_mutexed_object = instruction->mut_operand_type(args->string(i));
      rw_mutexed_object->Init<StringObject>(str);
    }
  }
  void Compute(Instruction* instruction) const override {
    // do nothing
  }
};
COMMAND(RegisterInstructionType<InitStringSymbolInstructionType>("InitStringSymbol"));

}  // namespace vm
}  // namespace oneflow
