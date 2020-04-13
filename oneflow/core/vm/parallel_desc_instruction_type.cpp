#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace vm {

class NewParallelDescSymbolInstructionType final : public InstructionType {
 public:
  NewParallelDescSymbolInstructionType() = default;
  ~NewParallelDescSymbolInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(ParallelDescObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, logical_object_id);
  FLAT_MSG_VIEW_END(ParallelDescObjectInstrOperand);
  // clang-format on

  void Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run<&GetTypeLogicalObjectId>(scheduler, instr_msg);
  }
  void Compute(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run<&GetSelfLogicalObjectId>(scheduler, instr_msg);
  }
  void Infer(InstrCtx*) const override { UNIMPLEMENTED(); }
  void Compute(InstrCtx*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(Scheduler* scheduler, InstructionMsg* instr_msg) const {
    FlatMsgView<ParallelDescObjectInstrOperand> view(instr_msg->operand());
    FOR_RANGE(int, i, 0, view->logical_object_id_size()) {
      int64_t logical_object_id = GetLogicalObjectId(view->logical_object_id(i));
      auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(
          scheduler->mut_scheduler_thread_only_allocator(), logical_object_id);
      CHECK(scheduler->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      auto mirrored_object = ObjectMsgPtr<MirroredObject>::NewFrom(scheduler->mut_allocator(),
                                                                   logical_object.Mutable(), 0);
      {
        const auto& serialized_conf =
            Global<Storage<ParallelConf>>::Get()->Get(view->logical_object_id(i));
        mirrored_object->Init<ObjectWrapper<ParallelDesc>>(serialized_conf);
      }
      CHECK(global_device_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
    }
  }
};
COMMAND(Global<Storage<ParallelConf>>::SetAllocated(new Storage<ParallelConf>()));
COMMAND(RegisterInstructionType<NewParallelDescSymbolInstructionType>("NewParallelDescSymbol"));
COMMAND(RegisterLocalInstructionType<NewParallelDescSymbolInstructionType>(
    "LocalNewParallelDescSymbol"));

}  // namespace vm
}  // namespace oneflow
