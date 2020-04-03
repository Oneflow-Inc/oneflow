#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/job_object.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

namespace {

std::shared_ptr<MemoryCase> MakeMemCase(const DeviceType device_type, const int64_t device_id) {
  auto mem_case = std::make_shared<MemoryCase>();
  if (device_type == DeviceType::kCPU) {
    mem_case->mutable_host_mem();
  } else if (device_type == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(device_id);
  } else {
    UNIMPLEMENTED();
  }
  return mem_case;
}

}  // namespace

class NewBlobObjectInstructionType final : public vm::InstructionType {
 public:
  NewBlobObjectInstructionType() = default;
  ~NewBlobObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewBlobObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::ConstOperand, job);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutableOperand, blob);
  FLAT_MSG_VIEW_END(NewBlobObjectInstrOperand);
  // clang-format on

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<NewBlobObjectInstrOperand> view;
    CHECK(view.Match(instr_ctx->instr_msg().operand()));
    const auto& job_object = instr_ctx->operand_type(view->job()).Get<JobObject>();
    auto mem_case = MakeMemCase(job_object.parallel_desc().device_type(),
                                instr_ctx->instr_chain().stream().thread_ctx().device_id());
    DataType data_type = job_object.job_desc().DefaultDataType();
    for (int i = 0; i < view->blob_size(); ++i) {
      CHECK_GT(view->blob(i).logical_object_id(), 0);
      instr_ctx->mut_operand_type(view->blob(i))->Mutable<BlobObject>(mem_case, data_type);
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override { TODO(); }
};
COMMAND(vm::RegisterInstructionType<NewBlobObjectInstructionType>("NewBlobObject"));
COMMAND(vm::RegisterLocalInstructionType<NewBlobObjectInstructionType>("NewLocalBlobObject"));

class DeleteBlobObjectInstructionType final : public vm::InstructionType {
 public:
  DeleteBlobObjectInstructionType() = default;
  ~DeleteBlobObjectInstructionType() override = default;

  using stream_type = vm::DeviceHelperStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteBlobObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutableOperand, blob);
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
COMMAND(vm::RegisterLocalInstructionType<DeleteBlobObjectInstructionType>("DeleteLocalBlobObject"));

}  // namespace eager
}  // namespace oneflow
