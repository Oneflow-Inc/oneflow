#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/control_vm_stream_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void VmScheduler::ReleaseVmInstruction(VmInstruction* vm_instruction,
                                       /*out*/ ReadyVmInstrCtxList* ready_vm_instruction_list) {
  auto* mirrored_object_accesses = vm_instruction->mut_mirrored_object2access();
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(mirrored_object_accesses, access) {
    mirrored_object_accesses->Erase(access);
    auto* mirrored_object = access->mirrored_object();
    mirrored_object->mut_access_list()->Erase(access);
    if (mirrored_object->access_list().empty()) {
      mirrored_object->logical_object().free_mirrored_object_handler().Call(
          mirrored_object->mut_logical_object());
    }
  }
  auto* out_edges = vm_instruction->mut_out_edges();
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(out_edges, out_edge) {
    ObjectMsgPtr<VmInstruction> out_vm_instr(out_edge->dst_vm_instr());
    out_vm_instr->mut_in_edges()->Erase(out_edge);
    if (out_vm_instr->in_edges().empty()) {
      ready_vm_instruction_list->EmplaceBack(std::move(out_vm_instr));
    }
    out_edges->Erase(out_edge);
  }
}

void VmScheduler::ReleaseVmInstructionPackage(
    VmInstructionPackage* pkg,
    /*out*/ ReadyVmInstrCtxList* ready_vm_instruction_list) {
  auto* vm_instruction_list = pkg->mut_vm_instruction_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instruction_list, vm_instruction) {
    ReleaseVmInstruction(vm_instruction, /*out*/ ready_vm_instruction_list);
    vm_instruction_list->Erase(vm_instruction);
  }
}

void VmScheduler::TryReleaseFinishedVmInstructionPackages(
    VmStream* vm_stream, /*out*/ ReadyVmInstrCtxList* ready_vm_instruction_list) {
  auto* running_pkg_list = vm_stream->mut_running_pkg_list();
  while (true) {
    auto* vm_instr_pkg = running_pkg_list->Begin();
    if (vm_instr_pkg == nullptr || !vm_instr_pkg->Done()) { break; }
    ReleaseVmInstructionPackage(vm_instr_pkg, /*out*/ ready_vm_instruction_list);
    vm_stream->DeleteVmInstructionPackage(vm_instr_pkg);
  }
}

void VmScheduler::FilterAndRunControlVmInstructions(TmpWaitingVmInstrMsgList* vm_instr_msg_list) {
  ControlVmStreamType control_vm_stream_type;
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    const VmStreamTypeId vm_stream_type_id =
        vm_instr_msg->vm_instruction_proto().vm_stream_type_id();
    if (vm_stream_type_id != kControlVmStreamTypeId) { continue; }
    control_vm_stream_type.Run(this, vm_instr_msg);
    vm_instr_msg_list->Erase(vm_instr_msg);
  }
}

void VmScheduler::MakeVmInstruction(TmpWaitingVmInstrMsgList* vm_instr_msg_list,
                                    /*out*/ NewVmInstrCtxList* ret_vm_instruction_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(vm_instr_msg_list, vm_instr_msg) {
    VmStreamTypeId vm_stream_type_id = vm_instr_msg->vm_instruction_proto().vm_stream_type_id();
    auto* vm_stream_rt_desc = mut_vm_stream_type_id2vm_stream_rt_desc()->FindPtr(vm_stream_type_id);
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(vm_stream_rt_desc->mut_parallel_id2vm_stream(),
                                            vm_stream) {
      auto vm_instruction =
          ObjectMsgPtr<VmInstruction>::NewFrom(mut_default_allocator(), vm_instr_msg, vm_stream);
      ret_vm_instruction_list->PushBack(vm_instruction.Mutable());
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
  vm_instruction->mut_mirrored_object2access()->Insert(mirrored_object_access.Mutable());
}

void VmScheduler::ConnectVmInstruction(VmInstruction* src_vm_instr, VmInstruction* dst_vm_instr) {
  auto edge =
      ObjectMsgPtr<VmInstrEdge>::NewFrom(mut_default_allocator(), src_vm_instr, dst_vm_instr);
  bool src_inserted = src_vm_instr->mut_out_edges()->Insert(edge.Mutable()).second;
  bool dst_inserted = dst_vm_instr->mut_in_edges()->Insert(edge.Mutable()).second;
  CHECK_EQ(src_inserted, dst_inserted);
}

void VmScheduler::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                         NewVmInstrCtxList* new_vm_instruction_list,
                                         /*out*/ ReadyVmInstrCtxList* ready_vm_instruction_list) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(new_vm_instruction_list, vm_instr) {
    int64_t parallel_id = vm_instr->vm_stream().vm_stream_id().parallel_id();
    const auto& operands = vm_instr->vm_instruction_msg().vm_instruction_proto().operand();
    for (const auto& operand : operands) {
      if (!operand.has_mutable_operand()) { continue; }
      auto* mirrored_object =
          FindMirroredObject(id2logical_object, operand.mutable_operand().value(), parallel_id);
      ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, vm_instr);
    }
    for (const auto& operand : operands) {
      if (!operand.has_const_operand()) { continue; }
      auto* mirrored_object =
          FindMirroredObject(id2logical_object, operand.const_operand().value(), parallel_id);
      ConsumeMirroredObject(kConstOperandAccess, mirrored_object, vm_instr);
    }
    auto* mirrored_object_accesses = vm_instr->mut_mirrored_object2access();
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(mirrored_object_accesses, mirrored_object_access) {
      auto* mirrored_object = mirrored_object_access->mirrored_object();
      if (mirrored_object->access_list().size() == 1) { continue; }
      if (mirrored_object_access->is_const_operand()) {
        auto* first = mirrored_object->mut_access_list()->Begin();
        if (!first->is_const_operand()) {
          ConnectVmInstruction(first->mut_vm_instruction(), vm_instr);
        }
      } else {
        auto* access_list = mirrored_object->mut_access_list();
        OBJECT_MSG_LIST_FOR_EACH_PTR(access_list, prev_access) {
          if (prev_access == mirrored_object_access) { break; }
          ConnectVmInstruction(prev_access->mut_vm_instruction(), vm_instr);
          access_list->Erase(prev_access);
        }
      }
    }
    if (vm_instr->in_edges().empty()) { ready_vm_instruction_list->PushBack(vm_instr); }
  }
}

void VmScheduler::DispatchVmInstruction(ReadyVmInstrCtxList* ready_vm_instruction_list) {
  OBJECT_MSG_LIST(VmStream, tmp_active_vm_stream_link) tmp_active_vm_stream_list;
  while (auto* first = ready_vm_instruction_list->Begin()) {
    auto* vm_stream = first->mut_vm_stream();
    ready_vm_instruction_list->MoveToDstBack(first, vm_stream->mut_collect_vm_instruction_list());
    if (vm_stream->is_tmp_active_vm_stream_link_empty()) {
      tmp_active_vm_stream_list.PushBack(vm_stream);
    }
  }
  auto* active_vm_stream_list = mut_active_vm_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_active_vm_stream_list, vm_stream) {
    tmp_active_vm_stream_list.Erase(vm_stream);
    auto pkg = vm_stream->NewVmInstructionPackage();
    vm_stream->mut_collect_vm_instruction_list()->MoveTo(pkg->mut_vm_instruction_list());
    vm_stream->mut_running_pkg_list()->PushBack(pkg.Mutable());
    if (vm_stream->is_active_vm_stream_link_empty()) { active_vm_stream_list->PushBack(vm_stream); }
    vm_stream->mut_waiting_pkg_list()->EmplaceBack(std::move(pkg));
  }
}

void VmScheduler::__Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator) {
  set_default_allocator(allocator);
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
  ReadyVmInstrCtxList ready_vm_instruction_list;
  auto* active_vm_stream_list = mut_active_vm_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(active_vm_stream_list, vm_stream) {
    TryReleaseFinishedVmInstructionPackages(vm_stream, /*out*/ &ready_vm_instruction_list);
    if (vm_stream->running_pkg_list().empty()) { active_vm_stream_list->Erase(vm_stream); }
  };
  auto* waiting_vm_instruction_list = mut_waiting_vm_instruction_list();
  if (waiting_msg_list().size() > 0) {
    TmpWaitingVmInstrMsgList tmp_waiting_msg_list;
    mut_waiting_msg_list()->MoveTo(&tmp_waiting_msg_list);
    FilterAndRunControlVmInstructions(&tmp_waiting_msg_list);
    NewVmInstrCtxList new_vm_instruction_list;
    MakeVmInstruction(&tmp_waiting_msg_list, /*out*/ &new_vm_instruction_list);
    ConsumeMirroredObjects(mut_id2logical_object(), &new_vm_instruction_list,
                           /*out*/ &ready_vm_instruction_list);
    new_vm_instruction_list.MoveTo(waiting_vm_instruction_list);
  }
  DispatchVmInstruction(&ready_vm_instruction_list);
}

}  // namespace oneflow
