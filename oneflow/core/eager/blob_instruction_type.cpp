#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

class DeleteBlobObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteBlobObjectInstructionType() = default;
  ~DeleteBlobObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteBlobObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, blob);
  FLAT_MSG_VIEW_END(DeleteBlobObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<DeleteBlobObjectInstrOperand> view;
    CHECK(view.Match(instr_ctx->instr_msg().operand()));
    for (int i = 0; i < view->blob_size(); ++i) {
      auto* type_mirrored_object = instr_ctx->mut_operand_type(view->blob(i));
      CHECK(type_mirrored_object->Has<BlobObject>());
      type_mirrored_object->reset_object();
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override { TODO(); }
};
COMMAND(vm::RegisterInstructionType<DeleteBlobObjectInstructionType>("DeleteBlobObject"));
COMMAND(vm::RegisterLocalInstructionType<DeleteBlobObjectInstructionType>("LocalDeleteBlobObject"));

}  // namespace eager
}  // namespace oneflow
