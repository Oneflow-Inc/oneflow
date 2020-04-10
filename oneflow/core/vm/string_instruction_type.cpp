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
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::InitConstHostOperand, string);
FLAT_MSG_VIEW_END(StringObjectInstrOperand);
// clang-format on

}  // namespace

class InitStringObjectInstructionType final : public InstructionType {
 public:
  InitStringObjectInstructionType() = default;
  ~InitStringObjectInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(InstrCtx* instr_ctx) const override {
    FlatMsgView<StringObjectInstrOperand> args(instr_ctx->instr_msg().operand());
    FOR_RANGE(int, i, 0, args->string_size()) {
      int64_t logical_object_id = args->string(i).logical_object_id();
      const auto& str = Global<Storage<std::string>>::Get()->Get(logical_object_id);
      auto* mirrored_object = instr_ctx->mut_operand_type(args->string(i));
      mirrored_object->Mutable<StringObject>(str);
    }
  }
  void Compute(InstrCtx* instr_ctx) const override {
    // do nothing
  }
};
COMMAND(RegisterInstructionType<InitStringObjectInstructionType>("InitStringObject"));
COMMAND(RegisterLocalInstructionType<InitStringObjectInstructionType>("LocalInitStringObject"));

}  // namespace vm
}  // namespace oneflow
