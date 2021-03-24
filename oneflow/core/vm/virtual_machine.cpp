/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace vm {

namespace {

bool IsSourceInstruction(const InstructionMsg& instr_msg) {
  for (const auto& instr_operand : instr_msg.operand()) {
    if (instr_operand->has_const_operand()) { return false; }
    if (instr_operand->has_mut_operand()) { return false; }
    if (instr_operand->has_mut2_operand()) { return false; }
    if (instr_operand->has_symbol_operand()) { return false; }
    if (instr_operand->has_init_symbol_operand()) { return false; }
    CHECK(instr_operand->has_separator() || instr_operand->has_double_operand()
          || instr_operand->has_double_operand() || instr_operand->has_int64_operand()
          || instr_operand->has_uint64_operand() || instr_operand->has_bool_operand());
  }
  return true;
}

}  // namespace

void VirtualMachine::ReleaseInstruction(Instruction* instruction,
                                        /*out*/ ReadyInstructionList* ready_instruction_list) {
  auto* rw_mutexed_object_accesses = instruction->mut_mirrored_object_id2access();
  OBJECT_MSG_SKIPLIST_FOR_EACH(rw_mutexed_object_accesses, access) {
    CHECK_GT(access->ref_cnt(), 1);
    rw_mutexed_object_accesses->Erase(access.Mutable());
    if (access->is_rw_mutexed_object_access_link_empty()) { continue; }
    auto* mirrored_object = access->mut_mirrored_object();
    mirrored_object->mut_rw_mutexed_object()->mut_access_list()->Erase(access.Mutable());
  }
  TryMoveWaitingToReady(instruction, ready_instruction_list, [](Instruction*) { return true; });
}

void VirtualMachine::TryReleaseFinishedInstructions(
    Stream* stream,
    /*out*/ ReadyInstructionList* ready_instruction_list) {
  auto* running_instruction_list = stream->mut_running_instruction_list();
  auto* sequantial_infer_list = mutable_sequantial_infer_instr_list();
  auto* sequantial_compute_list = mutable_sequantial_compute_instr_list();
  while (true) {
    auto* instruction_ptr = running_instruction_list->Begin();
    if (instruction_ptr == nullptr || !instruction_ptr->Done()) { break; }
    ReleaseInstruction(instruction_ptr, /*out*/ ready_instruction_list);
    const auto interpret_type = instruction_ptr->stream().stream_type_id().interpret_type();
    if (interpret_type == kInfer) {
      CHECK(!instruction_ptr->is_sequantial_infer_instr_link_empty());
      sequantial_infer_list->Erase(instruction_ptr);
    } else if (interpret_type == kCompute) {
      CHECK(!instruction_ptr->is_sequantial_compute_instr_link_empty());
      sequantial_compute_list->Erase(instruction_ptr);
    } else {
      UNIMPLEMENTED();
    }
    stream->DeleteInstruction(running_instruction_list->Erase(instruction_ptr));
  }
}

void VirtualMachine::FilterAndRunSourceInstructions(TmpPendingInstrMsgList* instr_msg_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(instr_msg_list, instr_msg) {
    const auto& instr_type_id = instr_msg->instr_type_id();
    const StreamType& stream_type = instr_type_id.stream_type_id().stream_type();
    if (stream_type.SharingVirtualMachineThread() && IsSourceInstruction(*instr_msg)) {
      const auto& parallel_desc = CHECK_JUST(GetInstructionParallelDesc(*instr_msg));
      if (!parallel_desc || parallel_desc->ContainingMachineId(this_machine_id())) {
        stream_type.Run(this, instr_msg);
      }
      instr_msg_list->Erase(instr_msg);
    }
  }
}

int64_t VirtualMachine::this_machine_id() const {
  CHECK_EQ(machine_id_range().size(), 1);
  return machine_id_range().begin();
}

void VirtualMachine::MakeInstructions(TmpPendingInstrMsgList* instr_msg_list,
                                      /*out*/ NewInstructionList* new_instruction_list) {
  auto IsStreamInParallelDesc = [](const ParallelDesc* parallel_desc, const Stream& stream) {
    if (parallel_desc == nullptr) { return true; }
    if (stream.stream_type().SharingVirtualMachineThread()) {
      return parallel_desc->ContainingMachineId(stream.machine_id());
    }
    return parallel_desc->Containing(stream.machine_id(), stream.device_id());
  };
  auto* sequantial_infer_list = mutable_sequantial_infer_instr_list();
  auto* sequantial_compute_list = mutable_sequantial_compute_instr_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(instr_msg_list, instr_msg) {
    const StreamTypeId& stream_type_id = instr_msg->instr_type_id().stream_type_id();
    auto* stream_rt_desc = mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
    const auto& instruction_type = instr_msg->instr_type_id().instruction_type();
    if (stream_rt_desc == nullptr) {
      LOG(FATAL) << typeid(instruction_type).name() << " "
                 << typeid(stream_type_id.stream_type()).name();
    }
    bool is_sequantial = instruction_type.IsSequantialInstructionType();
    if (is_sequantial) { CHECK_EQ(stream_rt_desc->stream_id2stream().size(), 1); }
    const auto& parallel_desc = CHECK_JUST(GetInstructionParallelDesc(*instr_msg));
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(stream_rt_desc->mut_stream_id2stream(), stream) {
      if (!IsStreamInParallelDesc(parallel_desc.get(), *stream)) { continue; }
      ObjectMsgPtr<Instruction> instr = stream->NewInstruction(instr_msg, parallel_desc);
      if (stream_type_id.interpret_type() == kInfer) {
        sequantial_infer_list->PushBack(instr.Mutable());
      } else if (stream_type_id.interpret_type() == kCompute) {
        sequantial_compute_list->PushBack(instr.Mutable());
      } else {
        UNIMPLEMENTED();
      }
      if (!is_sequantial) { new_instruction_list->PushBack(instr.Mutable()); }
    }
    instr_msg_list->Erase(instr_msg);
  }
}

Maybe<ParallelDesc> VirtualMachine::GetInstructionParallelDesc(const InstructionMsg& instr_msg) {
  static const std::shared_ptr<ParallelDesc> empty_ptr;
  if (!instr_msg.has_parallel_desc_symbol_id()) { return empty_ptr; }
  int64_t symbol_id = instr_msg.parallel_desc_symbol_id();
  auto* logical_object = mut_id2logical_object()->FindPtr(symbol_id);
  CHECK_NOTNULL_OR_RETURN(logical_object) << "symbol_id: " << symbol_id;
  auto* map = logical_object->mut_global_device_id2mirrored_object();
  CHECK_EQ_OR_RETURN(map->size(), 1);
  return JUST(map->Begin()->rw_mutexed_object().Get<ObjectWrapper<ParallelDesc>>()).GetPtr();
}

MirroredObject* VirtualMachine::MutMirroredObject(int64_t logical_object_id,
                                                  int64_t global_device_id) {
  auto* logical_object = mut_id2logical_object()->FindPtr(logical_object_id);
  if (logical_object == nullptr) { return nullptr; }
  return logical_object->mut_global_device_id2mirrored_object()->FindPtr(global_device_id);
}

const MirroredObject* VirtualMachine::GetMirroredObject(int64_t logical_object_id,
                                                        int64_t global_device_id) {
  return MutMirroredObject(logical_object_id, global_device_id);
}

template<int64_t (*TransformLogicalObjectId)(int64_t), typename DoEachT>
void VirtualMachine::ForEachMirroredObject(Id2LogicalObject* id2logical_object,
                                           const Operand& operand, int64_t global_device_id,
                                           const DoEachT& DoEach) {
  int64_t logical_object_id = operand.logical_object_id();
  logical_object_id = TransformLogicalObjectId(logical_object_id);
  auto* logical_object = id2logical_object->FindPtr(logical_object_id);
  if (logical_object == nullptr) { return; }
  auto* map = logical_object->mut_global_device_id2mirrored_object();
  if (operand.has_all_mirrored_object()) {
    OBJECT_MSG_MAP_FOR_EACH_PTR(map, mirrored_object) { DoEach(mirrored_object); }
  } else {
    auto* mirrored_object = map->FindPtr(operand.GetGlobalDeviceId(global_device_id));
    if (mirrored_object != nullptr) { DoEach(mirrored_object); }
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachConstMirroredObject(
    InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kConstModifier, mem_zone_modifier>& const_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = const_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
    ForEachMirroredObject<&IdUtil::GetValueId>(id2logical_object, operand, global_device_id,
                                               DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
  } else {
    UNIMPLEMENTED();
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachConstMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mutable_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    // do nothing
  } else {
    UNIMPLEMENTED();
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachMutMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mutable_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&IdUtil::GetValueId>(id2logical_object, operand, global_device_id,
                                               DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
  } else {
    UNIMPLEMENTED();
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachMutMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kTypeAndDataMutableModifier, mem_zone_modifier>& mut2_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mut2_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
    ForEachMirroredObject<&IdUtil::GetValueId>(id2logical_object, operand, global_device_id,
                                               DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
  } else {
    UNIMPLEMENTED();
  }
}

void VirtualMachine::ConsumeMirroredObject(OperandAccessType access_type,
                                           MirroredObject* mirrored_object,
                                           Instruction* instruction) {
  bool is_const_operand = (access_type == kConstOperandAccess);
  auto rw_mutexed_object_access = ObjectMsgPtr<RwMutexedObjectAccess>::NewFrom(
      instruction->mut_allocator(), instruction, mirrored_object, is_const_operand);
  bool success = instruction->mut_mirrored_object_id2access()
                     ->Insert(rw_mutexed_object_access.Mutable())
                     .second;
  if (success) {
    mirrored_object->mut_rw_mutexed_object()->mut_access_list()->EmplaceBack(
        std::move(rw_mutexed_object_access));
  }
}

void VirtualMachine::ConnectInstruction(Instruction* src_instruction,
                                        Instruction* dst_instruction) {
  CHECK_NE(src_instruction, dst_instruction);
  auto edge = ObjectMsgPtr<InstructionEdge>::NewFrom(mut_vm_thread_only_allocator(),
                                                     src_instruction, dst_instruction);
  bool src_inserted = src_instruction->mut_out_edges()->Insert(edge.Mutable()).second;
  bool dst_inserted = dst_instruction->mut_in_edges()->Insert(edge.Mutable()).second;
  CHECK_EQ(src_inserted, dst_inserted);
}

void VirtualMachine::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                            NewInstructionList* new_instruction_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_instruction_list, instruction) {
    int64_t global_device_id = instruction->stream().global_device_id();
    InterpretType interpret_type = instruction->stream().stream_type_id().interpret_type();
    auto ConsumeMutMirroredObject = [&](MirroredObject* mirrored_object) {
      ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, instruction);
    };
    auto ConsumeConstMirroredObject = [&](MirroredObject* mirrored_object) {
      ConsumeMirroredObject(kConstOperandAccess, mirrored_object, instruction);
    };
    const auto& operands = instruction->instr_msg().operand();
    for (const auto& operand : operands) {
      if (operand->has_mut_operand()) {
        ForEachMutMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                         operand->mut_operand(), global_device_id,
                                                         ConsumeMutMirroredObject);
      } else if (operand->has_mut2_operand()) {
        ForEachMutMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                         operand->mut2_operand(), global_device_id,
                                                         ConsumeMutMirroredObject);
      } else if (operand->has_init_symbol_operand()) {
        const auto& symbol_operand = operand->init_symbol_operand().operand();
        CHECK(symbol_operand.has_sole_mirrored_object());
        ForEachMutMirroredObject<kHostConstMemZoneModifier>(interpret_type, id2logical_object,
                                                            operand->init_symbol_operand(), 0,
                                                            ConsumeMutMirroredObject);
      } else {
        // do nothing
      }
    }
    for (const auto& operand : operands) {
      if (operand->has_const_operand()) {
        ForEachConstMirroredObject<kDeviceMemZoneModifier>(
            interpret_type, id2logical_object, operand->const_operand(), global_device_id,
            ConsumeConstMirroredObject);
      } else if (operand->has_mut_operand()) {
        ForEachConstMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                           operand->mut_operand(), global_device_id,
                                                           ConsumeConstMirroredObject);
      } else if (operand->has_symbol_operand()) {
        const auto& symbol_operand = operand->symbol_operand().operand();
        CHECK(symbol_operand.has_sole_mirrored_object());
        ForEachConstMirroredObject<kHostConstMemZoneModifier>(interpret_type, id2logical_object,
                                                              operand->symbol_operand(), 0,
                                                              ConsumeConstMirroredObject);
      } else if (operand->has_init_symbol_operand()) {
        const auto& symbol_operand = operand->init_symbol_operand().operand();
        CHECK(symbol_operand.has_sole_mirrored_object());
        ForEachConstMirroredObject<kHostConstMemZoneModifier>(interpret_type, id2logical_object,
                                                              operand->init_symbol_operand(), 0,
                                                              ConsumeConstMirroredObject);
      } else {
        // do nothing
      }
    }
    auto* rw_mutexed_object_accesses = instruction->mut_mirrored_object_id2access();
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(rw_mutexed_object_accesses, rw_mutexed_object_access) {
      auto* mirrored_object = rw_mutexed_object_access->mut_mirrored_object();
      if (mirrored_object->rw_mutexed_object().access_list().size() == 1) { continue; }
      if (rw_mutexed_object_access->is_const_operand()) {
        auto* first = mirrored_object->mut_rw_mutexed_object()->mut_access_list()->Begin();
        if (!first->is_const_operand() && first->mut_instruction() != instruction) {
          ConnectInstruction(first->mut_instruction(), instruction);
        }
      } else {
        auto* access_list = mirrored_object->mut_rw_mutexed_object()->mut_access_list();
        OBJECT_MSG_LIST_FOR_EACH_PTR(access_list, access) {
          if (access == rw_mutexed_object_access) { break; }
          if (access->mut_instruction() != instruction) {
            ConnectInstruction(access->mut_instruction(), instruction);
          }
          access_list->Erase(access);
        }
      }
    }
  }
}

void VirtualMachine::FilterReadyInstructions(NewInstructionList* new_instruction_list,
                                             /*out*/ ReadyInstructionList* ready_instruction_list) {
  OBJECT_MSG_LIST_FOR_EACH_PTR(new_instruction_list, instruction) {
    if (instruction->in_edges().empty()) {
      new_instruction_list->MoveToDstBack(instruction, ready_instruction_list);
    }
  }
}

void VirtualMachine::DispatchAndPrescheduleInstructions(
    ReadyInstructionList* ready_instruction_list) {
  PrescheduledInstructionList prescheduled;
  auto* active_stream_list = mut_active_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(ready_instruction_list, instruction) {
    auto* stream = instruction->mut_stream();
    ready_instruction_list->MoveToDstBack(instruction, stream->mut_running_instruction_list());
    if (stream->is_active_stream_link_empty()) { active_stream_list->PushBack(stream); }
    const auto& stream_type = stream->stream_type();
    if (stream_type.SharingVirtualMachineThread()) {
      stream_type.Run(this, instruction);
    } else {
      stream->mut_thread_ctx()->mut_pending_instruction_list()->PushBack(instruction);
    }
    TryMoveWaitingToReady(instruction, &prescheduled,
                          [stream](Instruction* dst) { return &dst->stream() == stream; });
  }
  prescheduled.MoveTo(ready_instruction_list);
}

template<typename ReadyList, typename IsEdgeReadyT>
void VirtualMachine::TryMoveWaitingToReady(Instruction* instruction, ReadyList* ready_list,
                                           const IsEdgeReadyT& IsEdgeReady) {
  auto* wait_instruction_list = mut_waiting_instruction_list();
  auto* out_edges = instruction->mut_out_edges();
  OBJECT_MSG_SKIPLIST_FOR_EACH_PTR(out_edges, out_edge) {
    Instruction* out_instruction = out_edge->dst_instruction();
    if (!IsEdgeReady(out_instruction)) { continue; }
    out_edges->Erase(out_edge);
    out_instruction->mut_in_edges()->Erase(out_edge);
    if (!out_instruction->in_edges().empty()) { continue; }
    wait_instruction_list->MoveToDstBack(out_instruction, ready_list);
  }
}

void VirtualMachine::__Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator) {
  mutable_vm_resource_desc()->CopyFrom(vm_desc.vm_resource_desc());
  CHECK_GT(vm_desc.machine_id_range().size(), 0);
  *mutable_machine_id_range() = vm_desc.machine_id_range();
  set_vm_thread_only_allocator(allocator);
  OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(&vm_desc.stream_type_id2desc(), stream_desc) {
    if (stream_desc->num_threads() == 0) { continue; }
    auto stream_rt_desc = ObjectMsgPtr<StreamRtDesc>::NewFrom(allocator, stream_desc);
    mut_stream_type_id2stream_rt_desc()->Insert(stream_rt_desc.Mutable());
    BalancedSplitter bs(stream_desc->parallel_num(), stream_desc->num_threads());
    for (int64_t i = 0, rel_global_device_id = 0; i < stream_desc->num_threads(); ++i) {
      auto thread_ctx = ObjectMsgPtr<ThreadCtx>::NewFrom(allocator, stream_rt_desc.Get());
      mut_thread_ctx_list()->PushBack(thread_ctx.Mutable());
      for (int j = bs.At(i).begin(); j < bs.At(i).end(); ++j, ++rel_global_device_id) {
        StreamId stream_id;
        stream_id.__Init__(stream_desc->stream_type_id(),
                           this_start_global_device_id() + rel_global_device_id);
        auto stream =
            ObjectMsgPtr<Stream>::NewFrom(mut_allocator(), thread_ctx.Mutable(), stream_id,
                                          vm_resource_desc().max_device_num_per_machine());
        CHECK(stream_rt_desc->mut_stream_id2stream()->Insert(stream.Mutable()).second);
        thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
      }
    }
  }
}

void VirtualMachine::Receive(InstructionMsgList* compute_instr_msg_list) {
  InstructionMsgList new_instr_msg_list;
  OBJECT_MSG_LIST_FOR_EACH_PTR(compute_instr_msg_list, compute_instr_msg) {
    new_instr_msg_list.EmplaceBack(compute_instr_msg->MakeInferInstrMsg());
    compute_instr_msg_list->MoveToDstBack(compute_instr_msg, &new_instr_msg_list);
  }
  mut_pending_msg_list()->MoveFrom(&new_instr_msg_list);
}

void VirtualMachine::Receive(ObjectMsgPtr<InstructionMsg>&& compute_instr_msg) {
  InstructionMsgList instr_msg_list;
  instr_msg_list.EmplaceBack(std::move(compute_instr_msg));
  Receive(&instr_msg_list);
}

namespace {

template<typename ContainerT>
void TryRunSequantialInstructionsOnList(VirtualMachine* vm, ContainerT* sequantial_list) {
  OBJECT_MSG_LIST_FOR_EACH(sequantial_list, instruction) {
    const auto& instruction_type = instruction->instr_msg().instr_type_id().instruction_type();
    if (!instruction_type.IsSequantialInstructionType()) { break; }
    sequantial_list->Erase(instruction.Mutable());
    const auto& stream_type = instruction->stream().stream_type();
    CHECK(stream_type.SharingVirtualMachineThread());
    stream_type.Run(vm, instruction.Mutable());
  }
}

}  // namespace

void VirtualMachine::TryRunSequantialInstructions() {
  TryRunSequantialInstructionsOnList(this, mutable_sequantial_infer_instr_list());
  TryRunSequantialInstructionsOnList(this, mutable_sequantial_compute_instr_list());
}

void VirtualMachine::Schedule() {
  ReadyInstructionList* ready_instruction_list = mut_ready_instruction_list();
  auto* active_stream_list = mut_active_stream_list();
  OBJECT_MSG_LIST_FOR_EACH_PTR(active_stream_list, stream) {
    TryReleaseFinishedInstructions(stream, /*out*/ ready_instruction_list);
    if (stream->running_instruction_list().empty()) { active_stream_list->Erase(stream); }
  }
  TryRunSequantialInstructions();
  auto* waiting_instruction_list = mut_waiting_instruction_list();
  if (pending_msg_list().size() > 0) {
    TmpPendingInstrMsgList tmp_pending_msg_list;
    mut_pending_msg_list()->MoveTo(&tmp_pending_msg_list);
    FilterAndRunSourceInstructions(&tmp_pending_msg_list);
    NewInstructionList new_instruction_list;
    MakeInstructions(&tmp_pending_msg_list, /*out*/ &new_instruction_list);
    ConsumeMirroredObjects(mut_id2logical_object(), &new_instruction_list);
    FilterReadyInstructions(&new_instruction_list, /*out*/ ready_instruction_list);
    new_instruction_list.MoveTo(waiting_instruction_list);
  }
  DispatchAndPrescheduleInstructions(ready_instruction_list);
}

bool VirtualMachine::Empty() const {
  return pending_msg_list().empty() && waiting_instruction_list().empty()
         && active_stream_list().empty();
}

}  // namespace vm
}  // namespace oneflow
