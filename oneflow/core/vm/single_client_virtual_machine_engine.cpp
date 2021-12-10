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

void VirtualMachineEngine::SingleClientSchedule() {
  // Release finished instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_active_stream_list()->size())) { ReleaseFinishedInstructions(); }
  // TODO(lixinqi): remove this line after disabling vm single-client support.
  if (unlikely(mut_delete_logical_object_list()->size())) { SingleClientTryDeleteLogicalObjects(); }
  // Try run the first barrier instruction.
  if (unlikely(mut_barrier_instruction_list()->size())) { TryRunBarrierInstruction(); }
  // Handle pending instructions, and try schedule them to ready list.
  // Use thread_unsafe_size to avoid acquiring mutex lock.
  // The inconsistency between pending_msg_list.list_head_.list_head_.container_ and
  // pending_msg_list.list_head_.list_head_.size_ is not a fatal error because
  // VirtualMachineEngine::Schedule is always in a buzy loop. All instructions will get handled
  // eventually.
  //  VirtualMachineEngine::Receive may be less effiencient if the thread safe version
  //  `pending_msg_list().size()` used here, because VirtualMachineEngine::Schedule is more likely
  //  to get the mutex lock.
  if (unlikely(pending_msg_list().thread_unsafe_size())) { SingleClientHandlePending(); }
  // dispatch ready instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_ready_instruction_list()->size())) { DispatchAndPrescheduleInstructions(); }
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

Maybe<const ParallelDesc> VirtualMachineEngine::SingleClientGetInstructionParallelDesc(
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

void VirtualMachineEngine::SingleClientRunInstructionInAdvance(InstructionMsg* instr_msg) {
  const auto& instr_type_id = instr_msg->instr_type_id();
  const StreamType& stream_type = instr_type_id.stream_type_id().stream_type();
  CHECK(stream_type.IsControlStreamType());
  CHECK(HasImmediateOperandsOnly(*instr_msg));
  const auto& parallel_desc = CHECK_JUST(SingleClientGetInstructionParallelDesc(*instr_msg));
  if (!parallel_desc || parallel_desc->ContainingMachineId(this_machine_id())) {
    stream_type.Run(this, instr_msg);
  }
}

void VirtualMachineEngine::SingleClientHandlePending() {
  OF_PROFILER_RANGE_PUSH("SingleClientHandlePending");
  InstructionMsgList tmp_pending_msg_list;
  // MoveTo is under a lock.
  mut_pending_msg_list()->MoveTo(&tmp_pending_msg_list);
  std::list<intrusive::shared_ptr<Instruction>> instructions;
  INTRUSIVE_FOR_EACH_PTR(instr_msg, &tmp_pending_msg_list) {
    if (instr_msg->instr_type_id().instruction_type().ResettingIdToObjectMap()) {
      SingleClientRunInstructionInAdvance(instr_msg);
    } else {
      SingleClientForEachNewInstruction(instr_msg, [&](Instruction* instruction) {
        instructions.push_back(intrusive::shared_ptr<Instruction>(instruction));
        SingleClientConsumeMirroredObjects(mut_id2logical_object(), instruction);
      });
    }
  }
  for (auto& instruction_elem : instructions) {
    auto* instruction = instruction_elem.Mutable();
    if (likely(Dispatchable(instruction))) { mut_ready_instruction_list()->PushBack(instruction); }
  }
  OF_PROFILER_RANGE_POP();
}

void VirtualMachineEngine::SingleClientTryDeleteLogicalObjects() {
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

namespace {

bool IsStreamInParallelDesc(const ParallelDesc* parallel_desc, const Stream& stream) {
  if (parallel_desc == nullptr) { return true; }
  if (stream.stream_type().IsControlStreamType()) {
    return parallel_desc->ContainingMachineId(stream.machine_id());
  }
  return parallel_desc->Containing(stream.machine_id(), stream.device_id());
}

}  // namespace

void VirtualMachineEngine::SingleClientForEachNewInstruction(
    InstructionMsg* instr_msg, Stream* stream, const std::shared_ptr<const ParallelDesc>& pd,
    const std::function<void(Instruction*)>& DoEachInstruction) {
  intrusive::shared_ptr<Instruction> instruction = stream->NewInstruction(instr_msg, pd);
  mut_lively_instruction_list()->PushBack(instruction.Mutable());
  const auto& instruction_type = instr_msg->instr_type_id().instruction_type();
  bool is_barrier_instruction = instruction_type.IsFrontSequential();
  if (unlikely(is_barrier_instruction)) {
    mut_barrier_instruction_list()->PushBack(instruction.Mutable());
  } else {
    DoEachInstruction(instruction.Mutable());
  }
}

void VirtualMachineEngine::SingleClientForEachNewInstruction(
    InstructionMsg* instr_msg, const std::function<void(Instruction*)>& DoEachInstruction) {
  const auto& instruction_type = instr_msg->instr_type_id().instruction_type();
  const StreamTypeId& stream_type_id = instr_msg->instr_type_id().stream_type_id();
  auto* stream_rt_desc = mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
  if (unlikely(stream_rt_desc == nullptr)) {
    const auto& stream_type = stream_type_id.stream_type();
    LOG(FATAL) << typeid(instruction_type).name() << " " << typeid(stream_type).name();
  }
  bool is_barrier_instruction = instruction_type.IsFrontSequential();
  if (unlikely(is_barrier_instruction)) { CHECK_EQ(stream_rt_desc->device_id2stream().size(), 1); }
  const auto& parallel_desc = CHECK_JUST(SingleClientGetInstructionParallelDesc(*instr_msg));
  for (const auto& stream : stream_rt_desc->device_id2stream()) {
    if (!IsStreamInParallelDesc(parallel_desc.get(), *stream)) { continue; }
    SingleClientForEachNewInstruction(instr_msg, stream.get(), parallel_desc, DoEachInstruction);
  }
}

void VirtualMachineEngine::SingleClientConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                                              Instruction* instruction) {
  const auto& ConsumeMirroredObject = [&](OperandAccessType access_type,
                                          MirroredObject* mirrored_object) {
    auto* access = AccessMirroredObject(access_type, mirrored_object, instruction);
    instruction->mut_mirrored_object_id2access()->Insert(access);
    return access;
  };
  auto ConsumeConstMirroredObject = [&](MirroredObject* mirrored_object) {
    ConsumeMirroredObject(kConstOperandAccess, mirrored_object);
  };
  auto ConsumeMutMirroredObject = [&](MirroredObject* mirrored_object) {
    ConsumeMirroredObject(kMutableOperandAccess, mirrored_object);
  };
  auto ConsumeDelMirroredObject = [&](MirroredObject* mirrored_object) {
    auto* access = ConsumeMirroredObject(kMutableOperandAccess, mirrored_object);
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
      ForEachConstMirroredObject<kDeviceMemZoneModifier>(interpret_type, id2logical_object,
                                                         operand->const_operand(), global_device_id,
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
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(rw_mutexed_object_access, rw_mutexed_object_accesses) {
    auto* mirrored_object = rw_mutexed_object_access->mut_mirrored_object();
    if (mirrored_object->has_deleting_access()
        && mirrored_object->mut_deleting_access() != rw_mutexed_object_access) {
      UNIMPLEMENTED() << " accessing a deleting object "
                      << mirrored_object->mirrored_object_id().logical_object_id_value();
    }
    if (mirrored_object->rw_mutexed_object().access_list().size() == 1) { continue; }
    if (rw_mutexed_object_access->is_const_operand()) {
      ConnectInstructionsByRead(rw_mutexed_object_access);
    } else {
      ConnectInstructionsByWrite(rw_mutexed_object_access);
    }
  }
}

}  // namespace vm
}  // namespace oneflow
