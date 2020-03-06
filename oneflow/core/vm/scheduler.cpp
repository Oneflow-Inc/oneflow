#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void VmScheduler::ReleaseVmInstruction(VmInstrChain* vm_instr_chain,
                                       /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_instr_chain->mut_vm_instruction_list(), vm_instruction) {
    auto* mirrored_object_accesses = vm_instruction->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(mirrored_object_accesses, access) {
      mirrored_object_accesses->Erase(access);
      auto* mirrored_object = access->mut_mirrored_object();
      mirrored_object->mut_access_list()->Erase(access);
      if (mirrored_object->access_list().empty()) {
        mirrored_object->logical_object().free_mirrored_object_handler().Call(
            mirrored_object->mut_logical_object());
      }
    }
  }
  auto* out_edges = vm_instr_chain->mut_out_edges();
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(out_edges, out_edge) {
    ObjectMsgPtr<VmInstrChain> out_vm_instr_chain(out_edge->dst_vm_instr_chain());
    out_vm_instr_chain->mut_in_edges()->Erase(out_edge);
    if (out_vm_instr_chain->in_edges().empty()) {
      ready_vm_instr_chain_list->EmplaceBack(std::move(out_vm_instr_chain));
    }
    out_edges->Erase(out_edge);
  }
}

void VmScheduler::ReleaseVmInstrChainPackage(
    VmInstrChainPackage* pkg,
    /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list) {
  auto* vm_instr_chain_list = pkg->mut_vm_instr_chain_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_chain_list, vm_instr_chain) {
    ReleaseVmInstruction(vm_instr_chain, /*out*/ ready_vm_instr_chain_list);
    vm_instr_chain_list->Erase(vm_instr_chain);
  }
}

void VmScheduler::TryReleaseFinishedVmInstrChainPackages(
    VmStream* vm_stream, /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list) {
  auto* running_pkg_list = vm_stream->mut_running_pkg_list();
  while (true) {
    auto* vm_instr_chain_pkg = running_pkg_list->Begin();
    if (vm_instr_chain_pkg == nullptr || !vm_instr_chain_pkg->Done()) { break; }
    ReleaseVmInstrChainPackage(vm_instr_chain_pkg, /*out*/ ready_vm_instr_chain_list);
    vm_stream->DeleteVmInstrChainPackage(vm_instr_chain_pkg);
  }
}

void VmScheduler::FilterAndRunControlVmInstructions(TmpWaitingVmInstrMsgList* vm_instr_msg_list) {
  ControlVmStreamType control_vm_stream_type;
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    const VmStreamTypeId vm_stream_type_id =
        vm_instr_msg->vm_instruction_proto().vm_stream_type_id();
    if (vm_stream_type_id != ControlVmStreamType::kVmStreamTypeId) { continue; }
    control_vm_stream_type.Run(this, vm_instr_msg);
    vm_instr_msg_list->Erase(vm_instr_msg);
  }
}

void VmScheduler::MakeVmInstruction(TmpWaitingVmInstrMsgList* vm_instr_msg_list,
                                    /*out*/ NewVmInstrChainList* ret_vm_instr_chain_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    VmStreamTypeId vm_stream_type_id = vm_instr_msg->vm_instruction_proto().vm_stream_type_id();
    auto* vm_stream_rt_desc = mut_vm_stream_type_id2vm_stream_rt_desc()->FindPtr(vm_stream_type_id);
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(vm_stream_rt_desc->mut_parallel_id2vm_stream(),
                                            vm_stream) {
      auto vm_instr_chain = ObjectMsgPtr<VmInstrChain>::NewFrom(
          mut_scheduler_thread_only_allocator(), vm_instr_msg, vm_stream);
      ret_vm_instr_chain_list->PushBack(vm_instr_chain.Mutable());
    }
    vm_instr_msg_list->Erase(vm_instr_msg);
  }
}

MirroredObject* VmScheduler::FindMirroredObject(Id2LogicalObject* id2logical_object,
                                                const LogicalObjectId& logical_object_id,
                                                int64_t parallel_id) {
  auto* logical_object = id2logical_object->FindPtr(logical_object_id);
  CHECK_NOTNULL(logical_object);
  auto* ret = logical_object->mut_parallel_id2mirrored_object()->FindPtr(parallel_id);
  CHECK_NOTNULL(ret);
  return ret;
}

void VmScheduler::ConsumeMirroredObject(OperandAccessType access_type,
                                        MirroredObject* mirrored_object,
                                        VmInstruction* vm_instruction) {
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

void VmScheduler::ConnectVmInstruction(VmInstrChain* src_vm_instr_chain,
                                       VmInstrChain* dst_vm_instr_chain) {
  auto edge = ObjectMsgPtr<VmInstrChainEdge>::NewFrom(mut_scheduler_thread_only_allocator(),
                                                      src_vm_instr_chain, dst_vm_instr_chain);
  bool src_inserted = src_vm_instr_chain->mut_out_edges()->Insert(edge.Mutable()).second;
  bool dst_inserted = dst_vm_instr_chain->mut_in_edges()->Insert(edge.Mutable()).second;
  CHECK_EQ(src_inserted, dst_inserted);
}

void VmScheduler::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                         NewVmInstrChainList* new_vm_instr_chain_list,
                                         /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list) {
  auto* begin = new_vm_instr_chain_list->Begin();
  if (begin != nullptr) { CHECK_EQ(begin->vm_instruction_list().size(), 1); }
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_vm_instr_chain_list, vm_instr_chain) {
    int64_t parallel_id = vm_instr_chain->vm_stream().vm_stream_id().parallel_id();
    CHECK_EQ(vm_instr_chain->vm_instruction_list().size(), 1);
    auto* vm_instruction = vm_instr_chain->mut_vm_instruction_list()->Begin();
    const auto& operands = vm_instruction->vm_instr_msg().vm_instruction_proto().operand();
    for (const auto& operand : operands) {
      LogicalObjectId logical_object_id = 0;
      if (operand.has_mutable_operand()) {
        logical_object_id = operand.mutable_operand().value();
      } else if (operand.has_mutable_local_operand()) {
        logical_object_id = operand.mutable_local_operand().value();
      } else {
        continue;
      }
      auto* mirrored_object = FindMirroredObject(id2logical_object, logical_object_id, parallel_id);
      ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, vm_instruction);
    }
    for (const auto& operand : operands) {
      LogicalObjectId logical_object_id = 0;
      if (operand.has_const_operand()) {
        logical_object_id = operand.const_operand().value();
      } else if (operand.has_const_local_operand()) {
        logical_object_id = operand.const_local_operand().value();
      } else {
        continue;
      }
      auto* mirrored_object = FindMirroredObject(id2logical_object, logical_object_id, parallel_id);
      ConsumeMirroredObject(kConstOperandAccess, mirrored_object, vm_instruction);
    }
    auto* mirrored_object_accesses = vm_instruction->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(mirrored_object_accesses, mirrored_object_access) {
      auto* mirrored_object = mirrored_object_access->mut_mirrored_object();
      if (mirrored_object->access_list().size() == 1) { continue; }
      if (mirrored_object_access->is_const_operand()) {
        auto* first = mirrored_object->mut_access_list()->Begin();
        if (!first->is_const_operand()) {
          ConnectVmInstruction(first->mut_vm_instruction()->mut_vm_instr_chain(), vm_instr_chain);
        }
      } else {
        auto* access_list = mirrored_object->mut_access_list();
        OBJECT_MSG_LIST_FOR_EACH_PTR(access_list, access) {
          if (access == mirrored_object_access) { break; }
          ConnectVmInstruction(access->mut_vm_instruction()->mut_vm_instr_chain(), vm_instr_chain);
          access_list->Erase(access);
        }
      }
    }
    if (vm_instr_chain->in_edges().empty()) {
      new_vm_instr_chain_list->MoveToDstBack(vm_instr_chain, ready_vm_instr_chain_list);
    }
  }
}

void VmScheduler::DispatchVmInstruction(ReadyVmInstrChainList* ready_vm_instr_chain_list) {
  OBJECT_MSG_LIST(VmStream, tmp_active_vm_stream_link) tmp_active_vm_stream_list;
  while (auto* first = ready_vm_instr_chain_list->Begin()) {
    auto* vm_stream = first->mut_vm_stream();
    ready_vm_instr_chain_list->MoveToDstBack(first, vm_stream->mut_collect_vm_instr_chain_list());
    if (vm_stream->is_tmp_active_vm_stream_link_empty()) {
      tmp_active_vm_stream_list.PushBack(vm_stream);
    }
  }
  auto* active_vm_stream_list = mut_active_vm_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_active_vm_stream_list, vm_stream) {
    tmp_active_vm_stream_list.Erase(vm_stream);
    auto pkg = vm_stream->NewVmInstrChainPackage();
    vm_stream->mut_collect_vm_instr_chain_list()->MoveTo(pkg->mut_vm_instr_chain_list());
    vm_stream->mut_running_pkg_list()->PushBack(pkg.Mutable());
    if (vm_stream->is_active_vm_stream_link_empty()) { active_vm_stream_list->PushBack(vm_stream); }
    vm_stream->mut_vm_thread()->mut_waiting_pkg_list()->EmplaceBack(std::move(pkg));
  }
}

void VmScheduler::__Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator) {
  set_scheduler_thread_only_allocator(allocator);
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(&vm_desc.vm_stream_type_id2desc(), vm_stream_desc) {
    auto vm_stream_rt_desc = ObjectMsgPtr<VmStreamRtDesc>::NewFrom(allocator, vm_stream_desc);
    mut_vm_stream_type_id2vm_stream_rt_desc()->Insert(vm_stream_rt_desc.Mutable());
    BalancedSplitter bs(vm_stream_desc->parallel_num(), vm_stream_desc->num_threads());
    for (int i = 0; i < vm_stream_desc->num_threads(); ++i) {
      auto vm_thread = ObjectMsgPtr<VmThread>::NewFrom(allocator, vm_stream_rt_desc.Get());
      mut_vm_thread_list()->PushBack(vm_thread.Mutable());
      for (int parallel_id = bs.At(i).begin(); parallel_id < bs.At(i).end(); ++parallel_id) {
        FlatMsg<VmStreamId> vm_stream_id;
        vm_stream_id->set_vm_stream_type_id(vm_stream_desc->vm_stream_type_id());
        vm_stream_id->set_parallel_id(parallel_id);
        auto vm_stream = ObjectMsgPtr<VmStream>::NewFrom(mut_allocator(), vm_thread.Mutable(),
                                                         vm_stream_id.Get());
        vm_stream_rt_desc->mut_parallel_id2vm_stream()->Insert(vm_stream.Mutable());
        vm_thread->mut_vm_stream_list()->PushBack(vm_stream.Mutable());
      }
    }
  }
}

void VmScheduler::Receive(VmInstructionMsgList* vm_instr_list) {
  mut_waiting_msg_list()->MoveFrom(vm_instr_list);
}

void VmScheduler::Schedule() {
  ReadyVmInstrChainList ready_vm_instr_chain_list;
  auto* active_vm_stream_list = mut_active_vm_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(active_vm_stream_list, vm_stream) {
    TryReleaseFinishedVmInstrChainPackages(vm_stream, /*out*/ &ready_vm_instr_chain_list);
    if (vm_stream->running_pkg_list().empty()) { active_vm_stream_list->Erase(vm_stream); }
  };
  auto* waiting_vm_instr_chain_list = mut_waiting_vm_instr_chain_list();
  if (waiting_msg_list().size() > 0) {
    TmpWaitingVmInstrMsgList tmp_waiting_msg_list;
    mut_waiting_msg_list()->MoveTo(&tmp_waiting_msg_list);
    FilterAndRunControlVmInstructions(&tmp_waiting_msg_list);
    NewVmInstrChainList new_vm_instr_chain_list;
    MakeVmInstruction(&tmp_waiting_msg_list, /*out*/ &new_vm_instr_chain_list);
    ConsumeMirroredObjects(mut_id2logical_object(), &new_vm_instr_chain_list,
                           /*out*/ &ready_vm_instr_chain_list);
    new_vm_instr_chain_list.MoveTo(waiting_vm_instr_chain_list);
  }
  DispatchVmInstruction(&ready_vm_instr_chain_list);
}

}  // namespace oneflow
