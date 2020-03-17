#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {
namespace vm {

void Scheduler::ReleaseInstruction(InstrChain* instr_chain,
                                   /*out*/ ReadyInstrChainList* ready_instr_chain_list) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instruction_list(), instruction) {
    auto* mirrored_object_accesses = instruction->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(mirrored_object_accesses, access) {
      mirrored_object_accesses->Erase(access);
      if (access->is_mirrored_object_access_link_empty()) { continue; }
      auto* mirrored_object = access->mut_mirrored_object();
      mirrored_object->mut_access_list()->Erase(access);
    }
  }
  auto* wait_instr_chain_list = mut_waiting_instr_chain_list();
  auto* out_edges = instr_chain->mut_out_edges();
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(out_edges, out_edge) {
    InstrChain* out_instr_chain = out_edge->dst_instr_chain();
    out_instr_chain->mut_in_edges()->Erase(out_edge);
    if (out_instr_chain->in_edges().empty()) {
      wait_instr_chain_list->MoveToDstBack(out_instr_chain, ready_instr_chain_list);
    }
    out_edges->Erase(out_edge);
  }
}

void Scheduler::TryReleaseFinishedInstrChains(Stream* stream,
                                              /*out*/ ReadyInstrChainList* ready_instr_chain_list) {
  auto* running_chain_list = stream->mut_running_chain_list();
  while (true) {
    auto* instr_chain_ptr = running_chain_list->Begin();
    if (instr_chain_ptr == nullptr || !instr_chain_ptr->Done()) { break; }
    ReleaseInstruction(instr_chain_ptr, /*out*/ ready_instr_chain_list);
    stream->DeleteInstrChain(running_chain_list->Erase(instr_chain_ptr));
  }
}

void Scheduler::FilterAndRunSourceControlInstructions(TmpPendingInstrMsgList* instr_msg_list) {
  ControlStreamType control_stream_type;
  OBJECT_MSG_LIST_FOR_EACH_PTR(instr_msg_list, instr_msg) {
    const auto& proto = instr_msg->instr_type_id();
    if (proto.stream_type_id() != ControlStreamType::kStreamTypeId) { continue; }
    if (!control_stream_type.IsSourceOpcode(proto.opcode())) { continue; }
    control_stream_type.Run(this, instr_msg);
    instr_msg_list->Erase(instr_msg);
  }
}

void Scheduler::MakeInstrChains(TmpPendingInstrMsgList* instr_msg_list,
                                /*out*/ NewInstrChainList* new_instr_chain_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(instr_msg_list, instr_msg) {
    StreamTypeId stream_type_id = instr_msg->instr_type_id().stream_type_id();
    auto* stream_rt_desc = mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(stream_rt_desc->mut_stream_id2stream(), stream) {
      new_instr_chain_list->EmplaceBack(stream->NewInstrChain(instr_msg));
    }
    instr_msg_list->Erase(instr_msg);
  }
}

template<typename DoEachT>
void Scheduler::ForEachMirroredObject(Id2LogicalObject* id2logical_object,
                                      const MirroredObjectOperand& mirrored_object_operand,
                                      int64_t parallel_id, const DoEachT& DoEach) {
  auto* logical_object = id2logical_object->FindPtr(mirrored_object_operand.logical_object_id());
  auto* map = logical_object->mut_parallel_id2mirrored_object();
  if (mirrored_object_operand.has_all_parallel_id()) {
    OBJECT_MSG_MAP_FOR_EACH_PTR(map, mirrored_object) { DoEach(mirrored_object); }
    return;
  }
  CHECK_NOTNULL(logical_object);
  auto* ret = map->FindPtr(mirrored_object_operand.GetParallelId(parallel_id));
  CHECK_NOTNULL(ret);
  DoEach(ret);
}

void Scheduler::ConsumeMirroredObject(OperandAccessType access_type,
                                      MirroredObject* mirrored_object, Instruction* instruction) {
  bool is_const_operand = (access_type == kConstOperandAccess);
  auto mirrored_object_access = ObjectMsgPtr<MirroredObjectAccess>::NewFrom(
      instruction->mut_allocator(), instruction, mirrored_object, is_const_operand);
  bool success =
      instruction->mut_mirrored_object_id2access()->Insert(mirrored_object_access.Mutable()).second;
  if (success) {
    mirrored_object->mut_access_list()->EmplaceBack(std::move(mirrored_object_access));
  }
}

void Scheduler::ConnectInstruction(InstrChain* src_instr_chain, InstrChain* dst_instr_chain) {
  auto edge = ObjectMsgPtr<InstrChainEdge>::NewFrom(mut_scheduler_thread_only_allocator(),
                                                    src_instr_chain, dst_instr_chain);
  bool src_inserted = src_instr_chain->mut_out_edges()->Insert(edge.Mutable()).second;
  bool dst_inserted = dst_instr_chain->mut_in_edges()->Insert(edge.Mutable()).second;
  CHECK_EQ(src_inserted, dst_inserted);
}

void Scheduler::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                       NewInstrChainList* new_instr_chain_list) {
  auto* begin = new_instr_chain_list->Begin();
  if (begin != nullptr) { CHECK_EQ(begin->instruction_list().size(), 1); }
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_instr_chain_list, instr_chain) {
    int64_t parallel_id = instr_chain->stream().stream_id().parallel_id();
    CHECK_EQ(instr_chain->instruction_list().size(), 1);
    auto* instruction = instr_chain->mut_instruction_list()->Begin();
    const auto& operands = instruction->instr_msg().operand();
    for (const auto& operand : operands) {
      if (!operand->has_mutable_operand()) { continue; }
      const auto& mirrored_object_operand = operand->mutable_operand().operand();
      ForEachMirroredObject(id2logical_object, mirrored_object_operand, parallel_id,
                            [&](MirroredObject* mirrored_object) {
                              ConsumeMirroredObject(kMutableOperandAccess, mirrored_object,
                                                    instruction);
                            });
    }
    for (const auto& operand : operands) {
      if (!operand->has_const_operand()) { continue; }
      const auto& mirrored_object_operand = operand->const_operand().operand();
      ForEachMirroredObject(id2logical_object, mirrored_object_operand, parallel_id,
                            [&](MirroredObject* mirrored_object) {
                              ConsumeMirroredObject(kConstOperandAccess, mirrored_object,
                                                    instruction);
                            });
    }
    auto* mirrored_object_accesses = instruction->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(mirrored_object_accesses, mirrored_object_access) {
      auto* mirrored_object = mirrored_object_access->mut_mirrored_object();
      if (mirrored_object->access_list().size() == 1) { continue; }
      if (mirrored_object_access->is_const_operand()) {
        auto* first = mirrored_object->mut_access_list()->Begin();
        if (!first->is_const_operand()) {
          ConnectInstruction(first->mut_instruction()->mut_instr_chain(), instr_chain);
        }
      } else {
        auto* access_list = mirrored_object->mut_access_list();
        OBJECT_MSG_LIST_FOR_EACH_PTR(access_list, access) {
          if (access == mirrored_object_access) { break; }
          ConnectInstruction(access->mut_instruction()->mut_instr_chain(), instr_chain);
          access_list->Erase(access);
        }
      }
    }
  }
}

void Scheduler::MergeChains(NewInstrChainList* new_instr_chain_list) {
  // TODO(lixinqi)
}

void Scheduler::FilterReadyChains(NewInstrChainList* new_instr_chain_list,
                                  /*out*/ ReadyInstrChainList* ready_instr_chain_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_instr_chain_list, instr_chain) {
    if (instr_chain->in_edges().empty()) {
      new_instr_chain_list->MoveToDstBack(instr_chain, ready_instr_chain_list);
    }
  }
}

void Scheduler::DispatchInstruction(ReadyInstrChainList* ready_chain_list) {
  auto* active_stream_list = mut_active_stream_list();
  ControlStreamType control_stream_type;
  OBJECT_MSG_LIST_FOR_EACH_PTR(ready_chain_list, instr_chain) {
    auto* stream = instr_chain->mut_stream();
    if (stream->stream_id().stream_type_id() == ControlStreamType::kStreamTypeId) {
      control_stream_type.Run(this, instr_chain);
    } else {
      ready_chain_list->MoveToDstBack(instr_chain, stream->mut_running_chain_list());
      if (stream->is_active_stream_link_empty()) { active_stream_list->PushBack(stream); }
      stream->mut_thread_ctx()->mut_pending_chain_list()->PushBack(instr_chain);
    }
  }
  ready_chain_list->Clear();
}

void Scheduler::__Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator) {
  set_scheduler_thread_only_allocator(allocator);
  auto Init = [&](StreamDesc* stream_desc) {
    auto stream_rt_desc = ObjectMsgPtr<StreamRtDesc>::NewFrom(allocator, stream_desc);
    mut_stream_type_id2stream_rt_desc()->Insert(stream_rt_desc.Mutable());
    BalancedSplitter bs(stream_desc->parallel_num(), stream_desc->num_threads());
    for (int64_t i = 0, parallel_id = 0; i < stream_desc->num_threads(); ++i) {
      auto thread_ctx = ObjectMsgPtr<ThreadCtx>::NewFrom(allocator, stream_rt_desc.Get(), i);
      mut_thread_ctx_list()->PushBack(thread_ctx.Mutable());
      for (int j = bs.At(i).begin(); j < bs.At(i).end(); ++j, ++parallel_id) {
        FlatMsg<StreamId> stream_id;
        stream_id->set_stream_type_id(stream_desc->stream_type_id());
        stream_id->set_parallel_id(parallel_id);
        auto stream =
            ObjectMsgPtr<Stream>::NewFrom(mut_allocator(), thread_ctx.Mutable(), stream_id.Get());
        CHECK(stream_rt_desc->mut_stream_id2stream()->Insert(stream.Mutable()).second);
        thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
      }
    }
  };
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(&vm_desc.stream_type_id2desc(), stream_desc) {
    CHECK_NE(stream_desc->stream_type_id(), ControlStreamType::kStreamTypeId);
    Init(stream_desc);
  }
  Init(ObjectMsgPtr<StreamDesc>::New(ControlStreamType::kStreamTypeId, 1, 1, 1).Mutable());
}

void Scheduler::Receive(InstructionMsgList* instr_list) {
  mut_pending_msg_list()->MoveFrom(instr_list);
}

void Scheduler::Receive(ObjectMsgPtr<InstructionMsg>&& instruction_msg) {
  mut_pending_msg_list()->EmplaceBack(std::move(instruction_msg));
}

void Scheduler::Schedule() {
  ReadyInstrChainList ready_instr_chain_list;
  auto* active_stream_list = mut_active_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(active_stream_list, stream) {
    TryReleaseFinishedInstrChains(stream, /*out*/ &ready_instr_chain_list);
    if (stream->running_chain_list().empty()) { active_stream_list->Erase(stream); }
  };
  auto* waiting_instr_chain_list = mut_waiting_instr_chain_list();
  if (pending_msg_list().size() > 0) {
    TmpPendingInstrMsgList tmp_pending_msg_list;
    mut_pending_msg_list()->MoveTo(&tmp_pending_msg_list);
    FilterAndRunSourceControlInstructions(&tmp_pending_msg_list);
    NewInstrChainList new_instr_chain_list;
    MakeInstrChains(&tmp_pending_msg_list, /*out*/ &new_instr_chain_list);
    ConsumeMirroredObjects(mut_id2logical_object(), &new_instr_chain_list);
    MergeChains(&new_instr_chain_list);
    FilterReadyChains(&new_instr_chain_list, /*out*/ &ready_instr_chain_list);
    new_instr_chain_list.MoveTo(waiting_instr_chain_list);
  }
  DispatchInstruction(&ready_instr_chain_list);
}

bool Scheduler::Empty() const {
  return pending_msg_list().empty() && waiting_instr_chain_list().empty()
         && active_stream_list().empty();
}

}  // namespace vm
}  // namespace oneflow
