#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace vm {

class NewObjectInstructionType final : public InstructionType {
 public:
  NewObjectInstructionType() = default;
  ~NewObjectInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewObjectInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(ConstHostOperand, parallel_desc);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, logical_object_id);
  FLAT_MSG_VIEW_END(NewObjectInstruction);
  // clang-format on

  void Infer(Scheduler* scheduler, InstrCtx* instr_ctx) const override {
    Run<&GetTypeLogicalObjectId>(scheduler, instr_ctx);
  }
  void Compute(Scheduler* scheduler, InstrCtx* instr_ctx) const override {
    Run<&GetSelfLogicalObjectId>(scheduler, instr_ctx);
  }
  void Infer(InstrCtx*) const override { UNIMPLEMENTED(); }
  void Compute(InstrCtx*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(Scheduler* scheduler, InstrCtx* instr_ctx) const {
    FlatMsgView<NewObjectInstruction> view(instr_ctx->instr_msg().operand());
    const auto& parallel_desc_obj =
        instr_ctx->operand_type(view->parallel_desc()).Get<ObjectWrapper<ParallelDesc>>();
    FOR_RANGE(int, i, 0, view->logical_object_id_size()) {
      int64_t logical_object_id = GetLogicalObjectId(view->logical_object_id(i));
      auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(
          scheduler->mut_scheduler_thread_only_allocator(), logical_object_id);
      CHECK(scheduler->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      FOR_RANGE(int64_t, parallel_id, 0, parallel_desc_obj->parallel_num()) {
        int64_t global_device_id =
            scheduler->vm_resource_desc().GetGlobalDeviceId(parallel_desc_obj.Get(), parallel_id);
        auto mirrored_object = ObjectMsgPtr<MirroredObject>::NewFrom(
            scheduler->mut_allocator(), logical_object.Mutable(), global_device_id);
        CHECK(global_device_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
      }
    }
  }
};
COMMAND(RegisterInstructionType<NewObjectInstructionType>("NewObject"));
COMMAND(RegisterLocalInstructionType<NewObjectInstructionType>("LocalNewObject"));

class DeleteObjectInstructionType final : public InstructionType {
 public:
  DeleteObjectInstructionType() = default;
  ~DeleteObjectInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteObjectInstruction);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(MutOperand, object);
  FLAT_MSG_VIEW_END(DeleteObjectInstruction);
  // clang-format on

  void Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    // do nothing, delete objects in Compute method
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
    FlatMsgView<DeleteObjectInstruction> view(instr_msg->operand());
    FOR_RANGE(int, i, 0, view->object_size()) {
      CHECK(view->object(i).operand().has_all_mirrored_object());
      int64_t logical_object_id = view->object(i).operand().logical_object_id();
      logical_object_id = GetLogicalObjectId(logical_object_id);
      auto* logical_object = scheduler->mut_id2logical_object()->FindPtr(logical_object_id);
      CHECK_NOTNULL(logical_object);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      OBJECT_MSG_MAP_FOR_EACH_PTR(global_device_id2mirrored_object, mirrored_object) {
        CHECK(!mirrored_object->has_object());
        global_device_id2mirrored_object->Erase(mirrored_object);
      }
      scheduler->mut_id2logical_object()->Erase(logical_object);
    }
  }
};
COMMAND(RegisterInstructionType<DeleteObjectInstructionType>("DeleteObject"));
COMMAND(RegisterLocalInstructionType<DeleteObjectInstructionType>("LocalDeleteObject"));

}  // namespace vm
}  // namespace oneflow
