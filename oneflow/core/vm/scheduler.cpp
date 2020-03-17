#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {
namespace vm {

void Scheduler::ReleaseInstruction(InstrChain* vm_instr_chain,
                                   /*out*/ ReadyInstrChainList* ready_vm_instr_chain_list) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_instr_chain->mut_vm_instruction_list(), vm_instruction) {
    auto* mirrored_object_accesses = vm_instruction->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(mirrored_object_accesses, access) {
      mirrored_object_accesses->Erase(access);
      if (access->is_mirrored_object_access_link_empty()) { continue; }
      auto* mirrored_object = access->mut_mirrored_object();
      mirrored_object->mut_access_list()->Erase(access);
    }
  }
  auto* wait_vm_instr_chain_list = mut_waiting_vm_instr_chain_list();
  auto* out_edges = vm_instr_chain->mut_out_edges();
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(out_edges, out_edge) {
    InstrChain* out_vm_instr_chain = out_edge->dst_vm_instr_chain();
    out_vm_instr_chain->mut_in_edges()->Erase(out_edge);
    if (out_vm_instr_chain->in_edges().empty()) {
      wait_vm_instr_chain_list->MoveToDstBack(out_vm_instr_chain, ready_vm_instr_chain_list);
    }
    out_edges->Erase(out_edge);
  }
}

void Scheduler::TryReleaseFinishedInstrChains(
    Stream* vm_stream, /*out*/ ReadyInstrChainList* ready_vm_instr_chain_list) {
  auto* running_chain_list = vm_stream->mut_running_chain_list();
  while (true) {
    auto* vm_instr_chain_ptr = running_chain_list->Begin();
    if (vm_instr_chain_ptr == nullptr || !vm_instr_chain_ptr->Done()) { break; }
    ReleaseInstruction(vm_instr_chain_ptr, /*out*/ ready_vm_instr_chain_list);
    vm_stream->DeleteInstrChain(running_chain_list->Erase(vm_instr_chain_ptr));
  }
}

void Scheduler::FilterAndRunSourceControlInstructions(TmpPendingInstrMsgList* vm_instr_msg_list) {
  ControlStreamType control_vm_stream_type;
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    const auto& proto = vm_instr_msg->vm_instr_id();
    if (proto.vm_stream_type_id() != ControlStreamType::kStreamTypeId) { continue; }
    if (!control_vm_stream_type.IsSourceOpcode(proto.opcode())) { continue; }
    control_vm_stream_type.Run(this, vm_instr_msg);
    vm_instr_msg_list->Erase(vm_instr_msg);
  }
}

void Scheduler::MakeInstrChains(TmpPendingInstrMsgList* vm_instr_msg_list,
                                /*out*/ NewInstrChainList* new_vm_instr_chain_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    StreamTypeId vm_stream_type_id = vm_instr_msg->vm_instr_id().vm_stream_type_id();
    auto* vm_stream_rt_desc = mut_vm_stream_type_id2vm_stream_rt_desc()->FindPtr(vm_stream_type_id);
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(vm_stream_rt_desc->mut_vm_stream_id2vm_stream(),
                                            vm_stream) {
      new_vm_instr_chain_list->EmplaceBack(vm_stream->NewInstrChain(vm_instr_msg));
    }
    vm_instr_msg_list->Erase(vm_instr_msg);
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
                                      MirroredObject* mirrored_object,
                                      Instruction* vm_instruction) {
  bool is_const_operand = (access_type == kConstOperandAccess);
  auto mirrored_object_access = ObjectMsgPtr<MirroredObjectAccess>::NewFrom(
      vm_instruction->mut_allocator(), vm_instruction, mirrored_object, is_const_operand);
  bool success = vm_instruction->mut_mirrored_object_id2access()
                     ->Insert(mirrored_object_access.Mutable())
                     .second;
  if (success) {
    mirrored_object->mut_access_list()->EmplaceBack(std::move(mirrored_object_access));
  }
}

void Scheduler::ConnectInstruction(InstrChain* src_vm_instr_chain, InstrChain* dst_vm_instr_chain) {
  auto edge = ObjectMsgPtr<InstrChainEdge>::NewFrom(mut_scheduler_thread_only_allocator(),
                                                    src_vm_instr_chain, dst_vm_instr_chain);
  bool src_inserted = src_vm_instr_chain->mut_out_edges()->Insert(edge.Mutable()).second;
  bool dst_inserted = dst_vm_instr_chain->mut_in_edges()->Insert(edge.Mutable()).second;
  CHECK_EQ(src_inserted, dst_inserted);
}

void Scheduler::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                       NewInstrChainList* new_vm_instr_chain_list) {
  auto* begin = new_vm_instr_chain_list->Begin();
  if (begin != nullptr) { CHECK_EQ(begin->vm_instruction_list().size(), 1); }
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_vm_instr_chain_list, vm_instr_chain) {
    int64_t parallel_id = vm_instr_chain->vm_stream().vm_stream_id().parallel_id();
    CHECK_EQ(vm_instr_chain->vm_instruction_list().size(), 1);
    auto* vm_instruction = vm_instr_chain->mut_vm_instruction_list()->Begin();
    const auto& operands = vm_instruction->vm_instr_msg().operand();
    for (const auto& operand : operands) {
      if (!operand->has_mutable_operand()) { continue; }
      const auto& mirrored_object_operand = operand->mutable_operand().operand();
      ForEachMirroredObject(id2logical_object, mirrored_object_operand, parallel_id,
                            [&](MirroredObject* mirrored_object) {
                              ConsumeMirroredObject(kMutableOperandAccess, mirrored_object,
                                                    vm_instruction);
                            });
    }
    for (const auto& operand : operands) {
      if (!operand->has_const_operand()) { continue; }
      const auto& mirrored_object_operand = operand->const_operand().operand();
      ForEachMirroredObject(id2logical_object, mirrored_object_operand, parallel_id,
                            [&](MirroredObject* mirrored_object) {
                              ConsumeMirroredObject(kConstOperandAccess, mirrored_object,
                                                    vm_instruction);
                            });
    }
    auto* mirrored_object_accesses = vm_instruction->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(mirrored_object_accesses, mirrored_object_access) {
      auto* mirrored_object = mirrored_object_access->mut_mirrored_object();
      if (mirrored_object->access_list().size() == 1) { continue; }
      if (mirrored_object_access->is_const_operand()) {
        auto* first = mirrored_object->mut_access_list()->Begin();
        if (!first->is_const_operand()) {
          ConnectInstruction(first->mut_vm_instruction()->mut_vm_instr_chain(), vm_instr_chain);
        }
      } else {
        auto* access_list = mirrored_object->mut_access_list();
        OBJECT_MSG_LIST_FOR_EACH_PTR(access_list, access) {
          if (access == mirrored_object_access) { break; }
          ConnectInstruction(access->mut_vm_instruction()->mut_vm_instr_chain(), vm_instr_chain);
          access_list->Erase(access);
        }
      }
    }
  }
}

void Scheduler::MergeChains(NewInstrChainList* new_vm_instr_chain_list) {
  // TODO(lixinqi)
}

void Scheduler::FilterReadyChains(NewInstrChainList* new_vm_instr_chain_list,
                                  /*out*/ ReadyInstrChainList* ready_vm_instr_chain_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_vm_instr_chain_list, vm_instr_chain) {
    if (vm_instr_chain->in_edges().empty()) {
      new_vm_instr_chain_list->MoveToDstBack(vm_instr_chain, ready_vm_instr_chain_list);
    }
  }
}

void Scheduler::DispatchInstruction(ReadyInstrChainList* ready_chain_list) {
  auto* active_vm_stream_list = mut_active_vm_stream_list();
  ControlStreamType control_vm_stream_type;
  OBJECT_MSG_LIST_FOR_EACH_PTR(ready_chain_list, vm_instr_chain) {
    auto* vm_stream = vm_instr_chain->mut_vm_stream();
    if (vm_stream->vm_stream_id().vm_stream_type_id() == ControlStreamType::kStreamTypeId) {
      control_vm_stream_type.Run(this, vm_instr_chain);
    } else {
      ready_chain_list->MoveToDstBack(vm_instr_chain, vm_stream->mut_running_chain_list());
      if (vm_stream->is_active_vm_stream_link_empty()) {
        active_vm_stream_list->PushBack(vm_stream);
      }
      vm_stream->mut_vm_thread()->mut_pending_chain_list()->PushBack(vm_instr_chain);
    }
  }
  ready_chain_list->Clear();
}

void Scheduler::__Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator) {
  set_scheduler_thread_only_allocator(allocator);
  auto Init = [&](StreamDesc* vm_stream_desc) {
    auto vm_stream_rt_desc = ObjectMsgPtr<StreamRtDesc>::NewFrom(allocator, vm_stream_desc);
    mut_vm_stream_type_id2vm_stream_rt_desc()->Insert(vm_stream_rt_desc.Mutable());
    BalancedSplitter bs(vm_stream_desc->parallel_num(), vm_stream_desc->num_threads());
    for (int64_t i = 0, parallel_id = 0; i < vm_stream_desc->num_threads(); ++i) {
      auto vm_thread = ObjectMsgPtr<Thread>::NewFrom(allocator, vm_stream_rt_desc.Get(), i);
      mut_vm_thread_list()->PushBack(vm_thread.Mutable());
      for (int j = bs.At(i).begin(); j < bs.At(i).end(); ++j, ++parallel_id) {
        FlatMsg<StreamId> vm_stream_id;
        vm_stream_id->set_vm_stream_type_id(vm_stream_desc->vm_stream_type_id());
        vm_stream_id->set_parallel_id(parallel_id);
        auto vm_stream =
            ObjectMsgPtr<Stream>::NewFrom(mut_allocator(), vm_thread.Mutable(), vm_stream_id.Get());
        CHECK(vm_stream_rt_desc->mut_vm_stream_id2vm_stream()->Insert(vm_stream.Mutable()).second);
        vm_thread->mut_vm_stream_list()->PushBack(vm_stream.Mutable());
      }
    }
  };
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(&vm_desc.vm_stream_type_id2desc(), vm_stream_desc) {
    CHECK_NE(vm_stream_desc->vm_stream_type_id(), ControlStreamType::kStreamTypeId);
    Init(vm_stream_desc);
  }
  Init(ObjectMsgPtr<StreamDesc>::New(ControlStreamType::kStreamTypeId, 1, 1, 1).Mutable());
}

void Scheduler::Receive(InstructionMsgList* vm_instr_list) {
  mut_pending_msg_list()->MoveFrom(vm_instr_list);
}

void Scheduler::Receive(ObjectMsgPtr<InstructionMsg>&& vm_instruction_msg) {
  mut_pending_msg_list()->EmplaceBack(std::move(vm_instruction_msg));
}

void Scheduler::Schedule() {
  ReadyInstrChainList ready_vm_instr_chain_list;
  auto* active_vm_stream_list = mut_active_vm_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(active_vm_stream_list, vm_stream) {
    TryReleaseFinishedInstrChains(vm_stream, /*out*/ &ready_vm_instr_chain_list);
    if (vm_stream->running_chain_list().empty()) { active_vm_stream_list->Erase(vm_stream); }
  };
  auto* waiting_vm_instr_chain_list = mut_waiting_vm_instr_chain_list();
  if (pending_msg_list().size() > 0) {
    TmpPendingInstrMsgList tmp_pending_msg_list;
    mut_pending_msg_list()->MoveTo(&tmp_pending_msg_list);
    FilterAndRunSourceControlInstructions(&tmp_pending_msg_list);
    NewInstrChainList new_vm_instr_chain_list;
    MakeInstrChains(&tmp_pending_msg_list, /*out*/ &new_vm_instr_chain_list);
    ConsumeMirroredObjects(mut_id2logical_object(), &new_vm_instr_chain_list);
    MergeChains(&new_vm_instr_chain_list);
    FilterReadyChains(&new_vm_instr_chain_list, /*out*/ &ready_vm_instr_chain_list);
    new_vm_instr_chain_list.MoveTo(waiting_vm_instr_chain_list);
  }
  DispatchInstruction(&ready_vm_instr_chain_list);
}

bool Scheduler::Empty() const {
  return pending_msg_list().empty() && waiting_vm_instr_chain_list().empty()
         && active_vm_stream_list().empty();
}

}  // namespace vm
}  // namespace oneflow
