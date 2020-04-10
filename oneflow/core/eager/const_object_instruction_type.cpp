#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/eager/object_wrapper.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(ConstObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::InitConstHostOperand, serialized_logical_object_id);
FLAT_MSG_VIEW_END(ConstObjectInstrOperand);
// clang-format on

}  // namespace

template<typename T, typename SerializedT>
class InitConstObjectInstructionType final : public vm::InstructionType {
 public:
  InitConstObjectInstructionType() = default;
  ~InitConstObjectInstructionType() override = default;

  using stream_type = vm::HostStreamType;

  void Infer(vm::InstrCtx* instr_ctx) const override {
    FlatMsgView<ConstObjectInstrOperand> args(instr_ctx->instr_msg().operand());
    FOR_RANGE(int, i, 0, args->serialized_logical_object_id_size()) {
      const auto& operand = args->serialized_logical_object_id(i);
      int64_t logical_object_id = operand.logical_object_id();
      const auto& serialized_conf = Global<vm::Storage<SerializedT>>::Get()->Get(logical_object_id);
      auto* mirrored_object = instr_ctx->mut_operand_type(operand);
      mirrored_object->Mutable<ObjectWrapper<T>>(serialized_conf);
    }
  }
  void Compute(vm::InstrCtx* instr_ctx) const override {
    // do nothing
  }
};

namespace {

COMMAND(Global<vm::Storage<ParallelConf>>::SetAllocated(new vm::Storage<ParallelConf>()));
using ParallelDescInstr = InitConstObjectInstructionType<ParallelDesc, ParallelConf>;
COMMAND(vm::RegisterInstructionType<ParallelDescInstr>("InitParallelDescObject"));
COMMAND(vm::RegisterLocalInstructionType<ParallelDescInstr>("LocalInitParallelDescObject"));

}  // namespace

namespace {

COMMAND(Global<vm::Storage<JobConfigProto>>::SetAllocated(new vm::Storage<JobConfigProto>()));
using JobDescInstr = InitConstObjectInstructionType<JobDesc, JobConfigProto>;
COMMAND(vm::RegisterInstructionType<JobDescInstr>("InitJobDescObject"));
COMMAND(vm::RegisterLocalInstructionType<JobDescInstr>("LocalInitJobDescObject"));

}  // namespace

namespace {

COMMAND(Global<vm::Storage<OperatorConf>>::SetAllocated(new vm::Storage<OperatorConf>()));
using OperatorConfInstr = InitConstObjectInstructionType<OperatorConf, OperatorConf>;
COMMAND(vm::RegisterInstructionType<OperatorConfInstr>("InitOperatorConfObject"));
COMMAND(vm::RegisterLocalInstructionType<OperatorConfInstr>("LocalInitOperatorConfObject"));

}  // namespace

}  // namespace eager
}  // namespace oneflow
