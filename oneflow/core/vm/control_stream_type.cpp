#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

template<>
class InferStreamType<ControlStreamType> final : public StreamType {
 public:
  InferStreamType() = default;
  ~InferStreamType() = default;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override {
    return ControlStreamType().InitInstructionStatus(stream, status_buffer);
  }
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override {
    return ControlStreamType().DeleteInstructionStatus(stream, status_buffer);
  }
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override {
    return ControlStreamType().QueryInstructionStatusDone(stream, status_buffer);
  }
  void Infer(InstrChain* instr_chain) const override { UNIMPLEMENTED(); }
  void Infer(Scheduler* scheduler, InstrChain* instr_chain) const override {
    ControlStreamType().Infer(scheduler, instr_chain);
  }
  void Infer(Scheduler* scheduler, InstructionMsg* instruction_msg) const override {
    ControlStreamType().Infer(scheduler, instruction_msg);
  }
  void Compute(InstrChain* instr_chain) const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void Compute(Scheduler*, InstructionMsg*) const override { LOG(FATAL) << "UNIMPLEMENTED"; }

  bool SharingSchedulerThread() const override { return true; }

  ObjectMsgPtr<StreamDesc> MakeRemoteStreamDesc(const Resource& resource,
                                                int64_t this_machine_id) const override {
    auto stream_desc = ControlStreamType().MakeRemoteStreamDesc(resource, this_machine_id);
    stream_desc->mut_stream_type_id()->CopyFrom(
        LookupInferStreamTypeId(stream_desc->stream_type_id()));
    return stream_desc;
  }
  ObjectMsgPtr<StreamDesc> MakeLocalStreamDesc(const Resource& resource) const override {
    auto stream_desc = ControlStreamType().MakeLocalStreamDesc(resource);
    stream_desc->mut_stream_type_id()->CopyFrom(
        LookupInferStreamTypeId(stream_desc->stream_type_id()));
    return stream_desc;
  }
};

class NewSymbolInstructionType final : public InstructionType {
 public:
  NewSymbolInstructionType() = default;
  ~NewSymbolInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewSymbolCtrlInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, logical_object_id);
    FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, parallel_num);
  FLAT_MSG_VIEW_END(NewSymbolCtrlInstruction);
  // clang-format on

  void Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run(scheduler, instr_msg, &GetTypeLogicalObjectId);
  }
  void Compute(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run(scheduler, instr_msg, &GetSelfLogicalObjectId);
  }
  void Infer(InstrCtx*) const override { UNIMPLEMENTED(); }
  void Compute(InstrCtx*) const override { UNIMPLEMENTED(); }

 private:
  template<typename GetLogicalObjectIdT>
  void Run(Scheduler* scheduler, InstructionMsg* instr_msg,
           const GetLogicalObjectIdT& GetLogicalObjectId) const {
    FlatMsgView<NewSymbolCtrlInstruction> view;
    CHECK(view->Match(instr_msg->mut_operand()));
    uint64_t logical_object_id = GetLogicalObjectId(view->logical_object_id());
    auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(
        scheduler->mut_scheduler_thread_only_allocator(), logical_object_id);
    CHECK(scheduler->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
    auto* parallel_id2mirrored_object = logical_object->mut_parallel_id2mirrored_object();
    for (int64_t i = 0; i < view->parallel_num(); ++i) {
      auto mirrored_object = ObjectMsgPtr<MirroredObject>::NewFrom(scheduler->mut_allocator(),
                                                                   logical_object.Mutable(), i);
      CHECK(parallel_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
    }
  }
};
COMMAND(RegisterInstructionType<NewSymbolInstructionType>("NewSymbol"));
COMMAND(RegisterLocalInstructionType<NewSymbolInstructionType>("LocalNewSymbol"));

class DeleteSymbolInstructionType final : public InstructionType {
 public:
  DeleteSymbolInstructionType() = default;
  ~DeleteSymbolInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(DeleteSymbolCtrlInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
  FLAT_MSG_VIEW_END(DeleteSymbolCtrlInstruction);
  // clang-format on

  void Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    // do nothing, delete symbol in Compute method
  }
  void Compute(Scheduler* scheduler, InstructionMsg* instr_msg) const override {
    Run(scheduler, instr_msg, &GetTypeLogicalObjectId);
    Run(scheduler, instr_msg, &GetSelfLogicalObjectId);
  }
  void Infer(InstrCtx*) const override { UNIMPLEMENTED(); }
  void Compute(InstrCtx*) const override { UNIMPLEMENTED(); }

 private:
  template<typename GetLogicalObjectIdT>
  void Run(Scheduler* scheduler, InstructionMsg* instr_msg,
           const GetLogicalObjectIdT& GetLogicalObjectId) const {
    FlatMsgView<DeleteSymbolCtrlInstruction> view;
    CHECK(view->Match(instr_msg->mut_operand()));
    uint64_t logical_object_id = view->mirrored_object_operand().operand().logical_object_id();
    logical_object_id = GetLogicalObjectId(logical_object_id);
    auto* logical_object = scheduler->mut_id2logical_object()->FindPtr(logical_object_id);
    CHECK_NOTNULL(logical_object);
    auto* parallel_id2mirrored_object = logical_object->mut_parallel_id2mirrored_object();
    for (int i = 0; i < parallel_id2mirrored_object->size(); ++i) {
      auto* mirrored_object = parallel_id2mirrored_object->FindPtr(i);
      CHECK(!mirrored_object->has_mirrored_object_type());
      parallel_id2mirrored_object->Erase(mirrored_object);
    }
    scheduler->mut_id2logical_object()->Erase(logical_object);
  }
};
COMMAND(RegisterInstructionType<DeleteSymbolInstructionType>("DeleteSymbol"));
COMMAND(RegisterLocalInstructionType<DeleteSymbolInstructionType>("LocalDeleteSymbol"));

void ControlStreamType::Infer(Scheduler* scheduler, InstructionMsg* instr_msg) const {
  const auto& instr_type_id = instr_msg->instr_type_id();
  CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kInfer);
  instr_type_id.instruction_type().Infer(scheduler, instr_msg);
}

void ControlStreamType::Infer(Scheduler* scheduler, InstrChain* instr_chain) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instr_ctx_list(), instr_ctx) {
    Infer(scheduler, instr_ctx->mut_instr_msg());
  }
  auto* status_buffer = instr_chain->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

void ControlStreamType::Compute(Scheduler* scheduler, InstructionMsg* instr_msg) const {
  const auto& instr_type_id = instr_msg->instr_type_id();
  CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
  instr_type_id.instruction_type().Compute(scheduler, instr_msg);
}

void ControlStreamType::Compute(Scheduler* scheduler, InstrChain* instr_chain) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instr_ctx_list(), instr_ctx) {
    Compute(scheduler, instr_ctx->mut_instr_msg());
  }
  auto* status_buffer = instr_chain->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

void ControlStreamType::InitInstructionStatus(const Stream& stream,
                                              InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void ControlStreamType::DeleteInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool ControlStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void ControlStreamType::Compute(InstrChain* instr_chain) const { UNIMPLEMENTED(); }

ObjectMsgPtr<StreamDesc> ControlStreamType::MakeRemoteStreamDesc(const Resource& resource,
                                                                 int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<ControlStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id);
  return ret;
}

ObjectMsgPtr<StreamDesc> ControlStreamType::MakeLocalStreamDesc(const Resource& resource) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<ControlStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(0);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
