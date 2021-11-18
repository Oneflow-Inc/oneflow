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
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/vm/vm_desc.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/spin_counter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/global.h"

namespace oneflow {
namespace vm {

namespace {

bool HasImmediateOperandsOnly(const InstructionMsg& instr_msg) {
  for (const auto& instr_operand : instr_msg.operand()) {
    if (instr_operand->has_const_operand()) { return false; }
    if (instr_operand->has_mut_operand()) { return false; }
    if (instr_operand->has_mut2_operand()) { return false; }
    if (instr_operand->has_del_operand()) { return false; }
    if (instr_operand->has_symbol_operand()) { return false; }
    if (instr_operand->has_init_symbol_operand()) { return false; }
    CHECK(instr_operand->has_separator() || instr_operand->has_double_operand()
          || instr_operand->has_double_operand() || instr_operand->has_int64_operand()
          || instr_operand->has_uint64_operand() || instr_operand->has_bool_operand());
  }
  return true;
}

}  // namespace

void VirtualMachineEngine::ReleaseInstruction(Instruction* instruction) {
  auto* access_list = instruction->mut_access_list();
  auto* rw_mutexed_object_accesses = instruction->mut_mirrored_object_id2access();
  INTRUSIVE_FOR_EACH(access, access_list) {
    CHECK_GT(access->ref_cnt(), 1);
    access_list->Erase(access.Mutable());
    if (access->is_mirrored_object_id_inserted()) {
      rw_mutexed_object_accesses->Erase(access.Mutable());
    }
    auto* mirrored_object = access->mut_mirrored_object();
    if (!access->rw_mutexed_object_access_hook().empty()) {
      CHECK_EQ(access->mut_rw_mutexed_object(), mirrored_object->mut_rw_mutexed_object());
      mirrored_object->mut_rw_mutexed_object()->mut_access_list()->Erase(access.Mutable());
    }
  }
  CHECK_EQ(rw_mutexed_object_accesses->size(), 0);
  auto* out_edges = instruction->mut_out_edges();
  INTRUSIVE_FOR_EACH_PTR(out_edge, out_edges) {
    Instruction* out_instruction = out_edge->mut_dst_instruction();
    // Edges are erased only if the instruction is completed.
    out_edges->Erase(out_edge);
    out_instruction->mut_in_edges()->Erase(out_edge);
    TryMoveFromWaitingToReady(out_instruction);
  }
}

// Handle pending instructions, schedule them to waiting list or ready list.
void VirtualMachineEngine::MovePendingToReadyOrWaiting() {
  OF_PROFILER_RANGE_PUSH("MovePendingToReadyOrWaiting");
  InstructionMsgList tmp_pending_msg_list;
  // MoveTo is under a lock.
  mut_pending_msg_list()->MoveTo(&tmp_pending_msg_list);
  InstructionList new_instruction_list;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(instr_msg, &tmp_pending_msg_list) {
    if (unlikely(instr_msg->instr_type_id().instruction_type().ResettingIdToObjectMap())) {
      RunInstructionsInAdvance(instr_msg);
    } else {
      MakeInstructions(instr_msg, /*out*/ &new_instruction_list);
    }
  }
  INTRUSIVE_FOR_EACH_PTR(instruction, &new_instruction_list) {
    ConsumeMirroredObjects(mut_id2logical_object(), instruction);
    if (likely(Dispatchable(instruction))) {
      mut_ready_instruction_list()->PushBack(instruction);
      new_instruction_list.Erase(instruction);
    }
  }
  new_instruction_list.MoveTo(mut_waiting_instruction_list());
  OF_PROFILER_RANGE_POP();
}

// Collect ready instructions onto ready_instruction_list_
void VirtualMachineEngine::ReleaseFinishedInstructions() {
  INTRUSIVE_FOR_EACH_PTR(stream, mut_active_stream_list()) {
    while (true) {
      auto* instruction_ptr = stream->mut_running_instruction_list()->Begin();
      if (instruction_ptr == nullptr || !instruction_ptr->Done()) { break; }
      OF_PROFILER_RANGE_PUSH("ReleaseFinishedInstructions");
      ReleaseInstruction(instruction_ptr);
      stream->mut_running_instruction_list()->Erase(instruction_ptr);
      stream->DeleteInstruction(mut_lively_instruction_list()->Erase(instruction_ptr));
      OF_PROFILER_RANGE_POP();
    }
    if (stream->running_instruction_list().empty()) { mut_active_stream_list()->Erase(stream); }
  }
}

void VirtualMachineEngine::RunInstructionsInAdvance(InstructionMsg* instr_msg) {
  const auto& instr_type_id = instr_msg->instr_type_id();
  const StreamType& stream_type = instr_type_id.stream_type_id().stream_type();
  CHECK(stream_type.IsControlStreamType());
  CHECK(HasImmediateOperandsOnly(*instr_msg));
  const auto& parallel_desc = CHECK_JUST(GetInstructionParallelDesc(*instr_msg));
  if (!parallel_desc || parallel_desc->ContainingMachineId(this_machine_id())) {
    stream_type.Run(this, instr_msg);
  }
}

int64_t VirtualMachineEngine::this_machine_id() const {
  CHECK_EQ(machine_id_range().size(), 1);
  return machine_id_range().begin();
}

namespace {

bool IsStreamInParallelDesc(const ParallelDesc* parallel_desc, const Stream& stream) {
  if (parallel_desc == nullptr) { return true; }
  if (stream.stream_type().IsControlStreamType()) {
    return parallel_desc->ContainingMachineId(stream.machine_id());
  }
  return parallel_desc->Containing(stream.machine_id(), stream.device_id());
}

}  // namespace

void VirtualMachineEngine::MakeInstructions(InstructionMsg* instr_msg,
                                            /*out*/ InstructionList* new_instruction_list) {
  const auto& instruction_type = instr_msg->instr_type_id().instruction_type();
  const StreamTypeId& stream_type_id = instr_msg->instr_type_id().stream_type_id();
  bool is_barrier_instruction = instruction_type.IsFrontSequential();
  const auto& NewAndMove = [&](Stream* stream, const std::shared_ptr<const ParallelDesc>& pd) {
    intrusive::shared_ptr<Instruction> instr = stream->NewInstruction(instr_msg, pd);
    mut_lively_instruction_list()->PushBack(instr.Mutable());
    if (unlikely(is_barrier_instruction)) {
      mut_barrier_instruction_list()->PushBack(instr.Mutable());
    } else {
      new_instruction_list->PushBack(instr.Mutable());
    }
  };
  if (likely(instr_msg->phy_instr_stream() != nullptr)) {
    NewAndMove(instr_msg->phy_instr_stream(), instr_msg->phy_instr_parallel_desc());
  } else {
    auto* stream_rt_desc = mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
    if (unlikely(stream_rt_desc == nullptr)) {
      const auto& stream_type = stream_type_id.stream_type();
      LOG(FATAL) << typeid(instruction_type).name() << " " << typeid(stream_type).name();
    }
    if (unlikely(is_barrier_instruction)) {
      CHECK_EQ(stream_rt_desc->device_id2stream().size(), 1);
    }
    const auto& parallel_desc = CHECK_JUST(GetInstructionParallelDesc(*instr_msg));
    for (const auto& stream : stream_rt_desc->device_id2stream()) {
      if (!IsStreamInParallelDesc(parallel_desc.get(), *stream)) { continue; }
      NewAndMove(stream.get(), parallel_desc);
    }
  }
}

Maybe<const ParallelDesc> VirtualMachineEngine::GetInstructionParallelDesc(
    const InstructionMsg& instr_msg) {
  if (instr_msg.phy_instr_parallel_desc()) { return instr_msg.phy_instr_parallel_desc(); }
  static const std::shared_ptr<const ParallelDesc> empty_ptr;
  if (!instr_msg.has_parallel_desc_symbol_id()) { return empty_ptr; }
  int64_t symbol_id = instr_msg.parallel_desc_symbol_id();
  auto* logical_object = mut_id2logical_object()->FindPtr(symbol_id);
  CHECK_NOTNULL_OR_RETURN(logical_object) << "symbol_id: " << symbol_id;
  auto* map = logical_object->mut_global_device_id2mirrored_object();
  CHECK_EQ_OR_RETURN(map->size(), 1);
  const std::shared_ptr<const ParallelDesc> parallel_desc =
      JUST(map->Begin()->rw_mutexed_object().Get<ObjectWrapper<ParallelDesc>>()).GetPtr();
  return parallel_desc;
}

MirroredObject* VirtualMachineEngine::MutMirroredObject(int64_t logical_object_id,
                                                        int64_t global_device_id) {
  auto* logical_object = mut_id2logical_object()->FindPtr(logical_object_id);
  if (logical_object == nullptr) { return nullptr; }
  return logical_object->mut_global_device_id2mirrored_object()->FindPtr(global_device_id);
}

const MirroredObject* VirtualMachineEngine::GetMirroredObject(int64_t logical_object_id,
                                                              int64_t global_device_id) {
  return MutMirroredObject(logical_object_id, global_device_id);
}

template<int64_t (*TransformLogicalObjectId)(int64_t), typename DoEachT>
void VirtualMachineEngine::ForEachMirroredObject(Id2LogicalObject* id2logical_object,
                                                 const Operand& operand, int64_t global_device_id,
                                                 const DoEachT& DoEach) {
  int64_t logical_object_id = operand.logical_object_id();
  logical_object_id = TransformLogicalObjectId(logical_object_id);
  auto* logical_object = id2logical_object->FindPtr(logical_object_id);
  if (logical_object == nullptr) { return; }
  auto* map = logical_object->mut_global_device_id2mirrored_object();
  if (operand.has_all_mirrored_object()) {
    INTRUSIVE_FOR_EACH_PTR(mirrored_object, map) { DoEach(mirrored_object); }
  } else {
    auto* mirrored_object = map->FindPtr(operand.GetGlobalDeviceId(global_device_id));
    if (mirrored_object != nullptr) { DoEach(mirrored_object); }
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachineEngine::ForEachConstMirroredObject(
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
void VirtualMachineEngine::ForEachConstMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mut_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mut_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    // Do nothing
  } else {
    UNIMPLEMENTED();
  }
}

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachineEngine::ForEachMutMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mut_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mut_operand.operand();
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
void VirtualMachineEngine::ForEachMutMirroredObject(
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

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachineEngine::ForEachMutMirroredObject(
    const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
    const ModifiedOperand<kDeleteModifier, mem_zone_modifier>& mut_operand,
    int64_t global_device_id, const DoEachT& DoEach) {
  const Operand& operand = mut_operand.operand();
  if (interpret_type == InterpretType::kCompute) {
    ForEachMirroredObject<&IdUtil::GetValueId>(id2logical_object, operand, global_device_id,
                                               DoEach);
  } else if (interpret_type == InterpretType::kInfer) {
    ForEachMirroredObject<&IdUtil::GetTypeId>(id2logical_object, operand, global_device_id, DoEach);
  } else {
    UNIMPLEMENTED();
  }
}

RwMutexedObjectAccess* VirtualMachineEngine::ConsumeMirroredObject(OperandAccessType access_type,
                                                                   MirroredObject* mirrored_object,
                                                                   Instruction* instruction) {
  auto rw_mutexed_object_access =
      intrusive::make_shared<RwMutexedObjectAccess>(instruction, mirrored_object, access_type);
  instruction->mut_mirrored_object_id2access()->Insert(rw_mutexed_object_access.Mutable());
  instruction->mut_access_list()->PushBack(rw_mutexed_object_access.Mutable());
  mirrored_object->mut_rw_mutexed_object()->mut_access_list()->EmplaceBack(
      std::move(rw_mutexed_object_access));
  return rw_mutexed_object_access.Mutable();
}

void VirtualMachineEngine::ConnectInstruction(Instruction* src_instruction,
                                              Instruction* dst_instruction) {
  CHECK_NE(src_instruction, dst_instruction);
  auto edge = intrusive::make_shared<InstructionEdge>(src_instruction, dst_instruction);
  src_instruction->mut_out_edges()->PushBack(edge.Mutable());
  dst_instruction->mut_in_edges()->PushBack(edge.Mutable());
}

void VirtualMachineEngine::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                                  Instruction* instruction) {
  const auto& phy_instr_operand = instruction->instr_msg().phy_instr_operand();
  auto ConsumeConstMirroredObject = [&](MirroredObject* mirrored_object) {
    ConsumeMirroredObject(kConstOperandAccess, mirrored_object, instruction);
  };
  auto ConsumeMutMirroredObject = [&](MirroredObject* mirrored_object) {
    ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, instruction);
  };
  if (likely(phy_instr_operand)) {
    phy_instr_operand->ForEachMut2MirroredObject(ConsumeMutMirroredObject);
    phy_instr_operand->ForEachMutMirroredObject(ConsumeMutMirroredObject);
    phy_instr_operand->ForEachConstMirroredObject(ConsumeConstMirroredObject);
  } else {
    auto ConsumeDelMirroredObject = [&](MirroredObject* mirrored_object) {
      auto* access = ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, instruction);
      CHECK(!mirrored_object->has_deleting_access());
      mirrored_object->set_deleting_access(access);
    };
    const InterpretType interpret_type = instruction->stream().stream_type_id().interpret_type();
    int64_t global_device_id = instruction->stream().global_device_id();
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
      } else if (operand->has_del_operand()) {
        ForEachMutMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                         operand->del_operand(), global_device_id,
                                                         ConsumeDelMirroredObject);
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
  }
  auto* rw_mutexed_object_accesses = instruction->mut_mirrored_object_id2access();
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(rw_mutexed_object_access, rw_mutexed_object_accesses) {
    auto* mirrored_object = rw_mutexed_object_access->mut_mirrored_object();
    if (mirrored_object->has_deleting_access()
        && mirrored_object->mut_deleting_access() != rw_mutexed_object_access) {
      UNIMPLEMENTED() << " accessing a deleting object "
                      << mirrored_object->mirrored_object_id().logical_object_id_value();
    }
    if (mirrored_object->rw_mutexed_object().access_list().size() == 1) { continue; }
    if (rw_mutexed_object_access->is_const_operand()) {
      auto* first = mirrored_object->mut_rw_mutexed_object()->mut_access_list()->Begin();
      if (first->is_const_operand()) {
        // do nothing
      } else if (first->is_mut_operand()) {
        if (first->mut_instruction() != instruction) {
          ConnectInstruction(first->mut_instruction(), instruction);
        }
      } else {
        UNIMPLEMENTED();
      }
    } else {
      CHECK(rw_mutexed_object_access->is_mut_operand());
      auto* access_list = mirrored_object->mut_rw_mutexed_object()->mut_access_list();
      INTRUSIVE_FOR_EACH_PTR(access, access_list) {
        if (access == rw_mutexed_object_access) { break; }
        CHECK(access->is_const_operand() || access->is_mut_operand())
            << "access type " << access->access_type() << " not supported";
        if (access->mut_instruction() != instruction) {
          ConnectInstruction(access->mut_instruction(), instruction);
        }
        CHECK_EQ(access->mut_rw_mutexed_object(), mirrored_object->mut_rw_mutexed_object());
        access_list->Erase(access);
      }
    }
  }
}

bool VirtualMachineEngine::Dispatchable(Instruction* instruction) const {
  if (!instruction->dispatched_instruction_hook().empty()) { return false; }
  const auto* stream = &instruction->stream();
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_in_edges()) {
    const auto& src_instruction = edge->src_instruction();
    if (!(&src_instruction.stream() == stream /* same stream*/
          && !src_instruction.dispatched_instruction_hook().empty() /* dispatched */)) {
      return false;
    }
  }
  return true;
}

// Dispatch ready instructions and put prescheduled instructions onto ready_instruction_list_.
void VirtualMachineEngine::DispatchAndPrescheduleInstructions() {
  OF_PROFILER_RANGE_PUSH("DispatchAndPrescheduleInstructions");
  ReadyInstructionList tmp_ready_instruction_list;
  mut_ready_instruction_list()->MoveTo(&tmp_ready_instruction_list);
  INTRUSIVE_FOR_EACH(instruction, &tmp_ready_instruction_list) {
    // Erases `instruction` from tmp_ready_instruction_list before dispatching, because
    // `instruction.dispatched_instruction_hook_` are used in DispatchInstruction.
    tmp_ready_instruction_list.Erase(instruction.Mutable());
    DispatchInstruction(instruction.Mutable());
    // preschedule instructions
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_out_edges()) {
      TryMoveFromWaitingToReady(edge->mut_dst_instruction());
    }
  }
  OF_PROFILER_RANGE_POP();
}

void VirtualMachineEngine::DispatchInstruction(Instruction* instruction) {
  OF_PROFILER_RANGE_PUSH(
      "D:" + instruction->instr_msg().instr_type_name() + ":"
      + instruction->instr_msg().instr_type_id().instruction_type().DebugOpTypeName(instruction));
  auto* stream = instruction->mut_stream();
  stream->mut_running_instruction_list()->PushBack(instruction);
  if (stream->active_stream_hook().empty()) { mut_active_stream_list()->PushBack(stream); }
  const auto& stream_type = stream->stream_type();
  if (OnSchedulerThread(stream_type)) {
    stream_type.Run(this, instruction);
  } else {
    stream->mut_thread_ctx()->mut_pending_instruction_list()->PushBack(instruction);
  }
  OF_PROFILER_RANGE_POP();
}

void VirtualMachineEngine::__Init__(const VmDesc& vm_desc) {
  mut_vm_resource_desc()->CopyFrom(vm_desc.vm_resource_desc());
  CHECK_GT(vm_desc.machine_id_range().size(), 0);
  *mut_machine_id_range() = vm_desc.machine_id_range();
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(stream_desc, &vm_desc.stream_type_id2desc()) {
    if (stream_desc->num_threads() == 0) { continue; }
    auto stream_rt_desc = intrusive::make_shared<StreamRtDesc>(stream_desc);
    mut_stream_type_id2stream_rt_desc()->Insert(stream_rt_desc.Mutable());
    BalancedSplitter bs(stream_desc->parallel_num(), stream_desc->num_threads());
    for (int64_t i = 0, rel_global_device_id = 0; i < stream_desc->num_threads(); ++i) {
      auto thread_ctx = intrusive::make_shared<ThreadCtx>(stream_rt_desc.Get());
      mut_thread_ctx_list()->PushBack(thread_ctx.Mutable());
      for (int j = bs.At(i).begin(); j < bs.At(i).end(); ++j, ++rel_global_device_id) {
        StreamId stream_id;
        stream_id.__Init__(stream_desc->stream_type_id(),
                           this_start_global_device_id() + rel_global_device_id);
        auto stream = intrusive::make_shared<Stream>(
            thread_ctx.Mutable(), stream_id, vm_resource_desc().max_device_num_per_machine());
        stream_rt_desc->add_stream(stream);
        thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
      }
    }
  }
}

void VirtualMachineEngine::GetCachedInstrTypeIdAndPhyInstrStream(const std::string& instr_type_name,
                                                                 int device_id,
                                                                 InstrTypeId* instr_type_id,
                                                                 Stream** stream) {
  auto* cache = &instr_type_name2rt_instr_type_id_;
  auto iter = cache->find(instr_type_name);
  if (unlikely(iter == cache->end())) {
    const auto& instr_type_id_val = LookupInstrTypeId(instr_type_name);
    const auto& stream_type_id = instr_type_id_val.stream_type_id();
    auto* stream_rt_desc = this->mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
    iter = cache->emplace(instr_type_name, RtInstrTypeId(instr_type_id_val, stream_rt_desc)).first;
  }
  instr_type_id->CopyFrom(iter->second.instr_type_id());
  *stream = iter->second.GetStream(device_id);
}

void VirtualMachineEngine::GetInstrTypeIdAndSoleStream(const std::string& instr_type_name,
                                                       InstrTypeId* instr_type_id,
                                                       Stream** stream) {
  instr_type_id->CopyFrom(LookupInstrTypeId(instr_type_name));
  const auto& stream_type_id = instr_type_id->stream_type_id();
  auto* stream_rt_desc = this->mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
  *stream = stream_rt_desc->GetSoleStream();
}

int64_t InstructionMaxRunningSeconds() { return 60 * 5; }

// Returns true if old pending_instruction_list is empty
Maybe<bool> VirtualMachineEngine::Receive(InstructionMsgList* compute_instr_msg_list) {
  OF_PROFILER_RANGE_PUSH("vm:Receive");
  InstructionMsgList new_instr_msg_list;
  INTRUSIVE_FOR_EACH_PTR(compute_instr_msg, compute_instr_msg_list) {
    if (!compute_instr_msg->phy_instr_operand()) {
      new_instr_msg_list.EmplaceBack(compute_instr_msg->MakeInferInstrMsg());
    }
    compute_instr_msg_list->MoveToDstBack(compute_instr_msg, &new_instr_msg_list);
  }
  const int64_t kHighWaterMark = GetInstructionHighWaterMark();
  const int64_t kLowWaterMark = GetInstructionLowWaterMark();
  if (flying_instruction_cnt() > kHighWaterMark) {
    JUST(Global<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
      const auto& NeedSpin = [&] { return flying_instruction_cnt() > kLowWaterMark; };
      while (true) {
        int64_t last_cnt = flying_instruction_cnt();
        const auto& ret = TRY(SpinWaitUntilTimeout(NeedSpin, InstructionMaxRunningSeconds()));
        if (ret.IsOk()) { break; }
        CHECK_NE_OR_RETURN(last_cnt, flying_instruction_cnt())
            << Error::UnimplementedError() << "The virtual machine don't respond in "
            << InstructionMaxRunningSeconds() << " seconds.";
      }
      return Maybe<void>::Ok();
    }));
  }
  bool old_list_empty = mut_pending_msg_list()->MoveFrom(&new_instr_msg_list);
  OF_PROFILER_RANGE_POP();
  return old_list_empty;
}

Maybe<bool> VirtualMachineEngine::Receive(
    intrusive::shared_ptr<InstructionMsg>&& compute_instr_msg) {
  InstructionMsgList instr_msg_list;
  instr_msg_list.EmplaceBack(std::move(compute_instr_msg));
  return Receive(&instr_msg_list);
}

bool VirtualMachineEngine::OnSchedulerThread(const StreamType& stream_type) {
  return stream_type.OnSchedulerThread() || pthread_fork::IsForkedSubProcess();
}

// Barrier instructions wait all non-barrier instructions to be done.
//
// `instruction.lively_instruction_hook_` is linked to `vm.lively_instruction_list_` for all
// instructions. `instruction.barrier_instruction_list_` is linked to `vm.barrier_instruction_list_`
// only for barrier instructions.
//
//
//  e.g. case0: waiting other instructions done.
//
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |      virtual_machine      |   |        instruction0       |   |        instruction1       |
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | lively_instruction_list_  |<->| lively_instruction_hook_  |<->| lively_instruction_hook_  |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | barrier_instruction_list_ |<+ | barrier_instruction_hook_ | +>| barrier_instruction_hook_ |
//  |---------------------------| | |---------------------------| | |---------------------------|
//  |            ...            | | |            ...            | | |            ...            |
//  +---------------------------+ | +---------------------------+ | +---------------------------+
//                                |                               |
//                                +-------------------------------+
//
// `instruction1` is a barrier instructions with barrier_instruction_hook_ linked, while
// instruction0 is not. From the `virtual_machine`'s view, `barrier_instruction_hook_.Begin() !=
// lively_instruction_hook_.Begin()`, so it's not the time to run barrier instruction
// `barrier_instruction_hook_.Begin()`.
//
//
//  e.g. case1: run barrier instructions.
//
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |      virtual_machine      |   |        instruction0       |   |        instruction1       |
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | lively_instruction_list_  |<->| lively_instruction_hook_  |<->| lively_instruction_hook_  |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | barrier_instruction_list_ |<->| barrier_instruction_hook_ |   | barrier_instruction_hook_ |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  |            ...            |   |            ...            |   |            ...            |
//  +---------------------------+   +---------------------------+   +---------------------------+
//
// `instruction0` is a barrier instructions with barrier_instruction_hook_ linked.
// From the `virtual_machine`'s view, `barrier_instruction_hook_.Begin() ==
// lively_instruction_hook_.Begin()`, so it's the time to run barrier instruction
// `barrier_instruction_hook_.Begin()`.
//
//
// With the introduction of barrier_instruction_list_/barrier_instruction_hook_, the function
// VirtualMachineEngine::Schedule can achive higher performance. For the most cases, barrier
// instructions are scarcely received by vm, there is no need for vm to run
// VirtualMachineEngine::TryRunBarrierInstruction every time VirtualMachineEngine::Schedule run. On
// the other hand, `barrier_instruction_hook_.size() == 0` is more lightweight than
// `lively_instruction_hook_.Begin()?->instr_msg().instr_type_id().instruction_type().IsFrontSequential()`
//
void VirtualMachineEngine::TryRunBarrierInstruction() {
  auto* sequnential_instruction = mut_barrier_instruction_list()->Begin();
  CHECK_NOTNULL(sequnential_instruction);
  if (likely(sequnential_instruction != mut_lively_instruction_list()->Begin())) { return; }
  // All instructions before `sequnential_instruction` are handled now, it's time to handle
  // `sequnential_instruction`.
  OF_PROFILER_RANGE_PUSH("RunBarrierInstruction");
  const auto& instr_type_id = sequnential_instruction->instr_msg().instr_type_id();
  const auto& instruction_type = instr_type_id.instruction_type();
  CHECK(instruction_type.IsFrontSequential());
  const StreamType& stream_type = instr_type_id.stream_type_id().stream_type();
  CHECK(OnSchedulerThread(stream_type));
  stream_type.Run(this, sequnential_instruction);
  mut_barrier_instruction_list()->Erase(sequnential_instruction);
  mut_lively_instruction_list()->Erase(sequnential_instruction);
  OF_PROFILER_RANGE_POP();
}

void VirtualMachineEngine::TryDeleteLogicalObjects() {
  auto* delete_list = mut_delete_logical_object_list();
  // INTRUSIVE_FOR_EACH_PTR supports removing elements at the end of iteration code
  INTRUSIVE_FOR_EACH_PTR(logical_object, delete_list) {
    auto* global_device_id2mirrored_object = logical_object->mut_global_device_id2mirrored_object();
    INTRUSIVE_FOR_EACH_PTR(mirrored_object, global_device_id2mirrored_object) {
      CHECK_EQ(mirrored_object->ref_cnt(), 1);
      if (mirrored_object->rw_mutexed_object().ref_cnt() == 1) {
        CHECK_EQ(mirrored_object->rw_mutexed_object().access_list().size(), 0);
        // TODO(lixinqi) fix the bug occured when uncommenting the next line
        // CHECK(!mirrored_object->rw_mutexed_object().has_object());
      }
      // `mirrored_object' is deleted by erasing
      global_device_id2mirrored_object->Erase(mirrored_object);
    }
    mut_id2logical_object()->Erase(logical_object);
    CHECK_EQ(logical_object->ref_cnt(), 1);
    // `logical_object' is deleted by erasing
    delete_list->Erase(logical_object);
  }
}

void VirtualMachineEngine::TryMoveFromWaitingToReady(Instruction* instruction) {
  if (Dispatchable(instruction)) {
    // For memory safety, do not swap the following two lines.
    mut_ready_instruction_list()->PushBack(instruction);
    mut_waiting_instruction_list()->Erase(instruction);
  }
}

void VirtualMachineEngine::Schedule() {
  // Release finished instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_active_stream_list()->size())) { ReleaseFinishedInstructions(); }
  // TODO(lixinqi): remove this line after disabling vm single-client support.
  if (unlikely(mut_delete_logical_object_list()->size())) { TryDeleteLogicalObjects(); }
  // Try run the first barrier instruction.
  if (unlikely(mut_barrier_instruction_list()->size())) { TryRunBarrierInstruction(); }
  // Handle pending instructions, schedule them to waiting list or ready list.
  // Use thread_unsafe_size to avoid acquiring mutex lock.
  // The inconsistency between pending_msg_list.list_head_.list_head_.container_ and
  // pending_msg_list.list_head_.list_head_.size_ is not a fatal error because
  // VirtualMachineEngine::Schedule is always in a buzy loop. All instructions will get handled
  // eventually.
  //  VirtualMachineEngine::Receive may be less effiencient if the thread safe version
  //  `pending_msg_list().size()` used here, because VirtualMachineEngine::Schedule is more likely
  //  to get the mutex lock.
  if (unlikely(pending_msg_list().thread_unsafe_size())) { MovePendingToReadyOrWaiting(); }
  // dispatch ready instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_ready_instruction_list()->size())) { DispatchAndPrescheduleInstructions(); }
}

bool VirtualMachineEngine::ThreadUnsafeEmpty() const {
  return active_stream_list().empty() && flying_instruction_cnt() == 0;
}

bool VirtualMachineEngine::Empty() const {
  // hook and size will be check in pending_msg_list().empty().
  return pending_msg_list().empty() && ThreadUnsafeEmpty();
}

}  // namespace vm
}  // namespace oneflow
