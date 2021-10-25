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
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/vm_desc.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/spin_counter.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/profiler/profiler.h"

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

void VirtualMachine::ReleaseInstruction(Instruction* instruction) {
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

// Collect ready instructions onto ready_instruction_list_
void VirtualMachine::TryReleaseFinishedInstructions(Stream* stream) {
  auto* running_instruction_list = stream->mut_running_instruction_list();
  auto* front_seq_compute_list = mut_front_seq_compute_instr_list();
  auto* vm_stat_running_list = mut_vm_stat_running_instruction_list();
  while (true) {
    auto* instruction_ptr = running_instruction_list->Begin();
    if (instruction_ptr == nullptr || !instruction_ptr->Done()) { break; }
    ReleaseInstruction(instruction_ptr);
    const auto interpret_type = instruction_ptr->stream().stream_type_id().interpret_type();
    if (interpret_type == kInfer) {
      // do nothing
    } else if (interpret_type == kCompute) {
      CHECK(!instruction_ptr->front_seq_compute_instr_hook().empty());
      front_seq_compute_list->Erase(instruction_ptr);
    } else {
      UNIMPLEMENTED();
    }
    vm_stat_running_list->Erase(instruction_ptr);
    stream->DeleteInstruction(running_instruction_list->Erase(instruction_ptr));
  }
}

void VirtualMachine::FilterAndRunInstructionsInAdvance(TmpPendingInstrMsgList* instr_msg_list) {
  INTRUSIVE_FOR_EACH_PTR(instr_msg, instr_msg_list) {
    const auto& instr_type_id = instr_msg->instr_type_id();
    if (instr_type_id.instruction_type().ResettingIdToObjectMap()) {
      const StreamType& stream_type = instr_type_id.stream_type_id().stream_type();
      CHECK(stream_type.IsControlStreamType());
      CHECK(HasImmediateOperandsOnly(*instr_msg));
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

namespace {

bool IsStreamInParallelDesc(const ParallelDesc* parallel_desc, const Stream& stream) {
  if (parallel_desc == nullptr) { return true; }
  if (stream.stream_type().IsControlStreamType()) {
    return parallel_desc->ContainingMachineId(stream.machine_id());
  }
  return parallel_desc->Containing(stream.machine_id(), stream.device_id());
}

}  // namespace

void VirtualMachine::MakeInstructions(TmpPendingInstrMsgList* instr_msg_list,
                                      /*out*/ NewInstructionList* new_instruction_list) {
  auto* front_seq_compute_list = mut_front_seq_compute_instr_list();
  INTRUSIVE_FOR_EACH_PTR(instr_msg, instr_msg_list) {
    const StreamTypeId& stream_type_id = instr_msg->instr_type_id().stream_type_id();
    auto* stream_rt_desc = mut_stream_type_id2stream_rt_desc()->FindPtr(stream_type_id);
    const auto& instruction_type = instr_msg->instr_type_id().instruction_type();
    if (stream_rt_desc == nullptr) {
      const auto& stream_type = stream_type_id.stream_type();
      LOG(FATAL) << typeid(instruction_type).name() << " " << typeid(stream_type).name();
    }
    bool is_front_seq = instruction_type.IsFrontSequential();
    if (is_front_seq) { CHECK_EQ(stream_rt_desc->stream_id2stream().size(), 1); }
    const auto& parallel_desc = CHECK_JUST(GetInstructionParallelDesc(*instr_msg));
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(stream, stream_rt_desc->mut_stream_id2stream()) {
      if (!IsStreamInParallelDesc(parallel_desc.get(), *stream)) { continue; }
      intrusive::shared_ptr<Instruction> instr = stream->NewInstruction(instr_msg, parallel_desc);
      if (stream_type_id.interpret_type() == kInfer) {
        // do nothing
      } else if (stream_type_id.interpret_type() == kCompute) {
        front_seq_compute_list->PushBack(instr.Mutable());
      } else {
        UNIMPLEMENTED();
      }
      if (!is_front_seq) { new_instruction_list->PushBack(instr.Mutable()); }
    }
    instr_msg_list->Erase(instr_msg);
  }
}

Maybe<const ParallelDesc> VirtualMachine::GetInstructionParallelDesc(
    const InstructionMsg& instr_msg) {
  static const std::shared_ptr<const ParallelDesc> empty_ptr;
  if (instr_msg.parallel_desc()) { return instr_msg.parallel_desc(); }
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
    INTRUSIVE_FOR_EACH_PTR(mirrored_object, map) { DoEach(mirrored_object); }
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

namespace {

template<typename CallbackT>
void ForEachConstMirroredObject4ConstPhyInstrOperand(InterpretType interpret_type,
                                                     const PhyInstrOperand& phy_instr_operand,
                                                     const CallbackT& Callback) {
  if (interpret_type == InterpretType::kCompute) {
    phy_instr_operand.ForEachConstMirroredObject(
        [&](MirroredObject* infer, MirroredObject* compute) {
          if (infer != nullptr) { Callback(infer); }
          if (compute != nullptr) { Callback(compute); }
        });
  } else if (interpret_type == InterpretType::kInfer) {
    phy_instr_operand.ForEachConstMirroredObject(
        [&](MirroredObject* infer, MirroredObject* compute) {
          if (infer != nullptr) { Callback(infer); }
        });
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachConstMirroredObject(
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

namespace {

template<typename CallbackT>
void ForEachConstMirroredObject4MutPhyInstrOperand(InterpretType interpret_type,
                                                   const PhyInstrOperand& phy_instr_operand,
                                                   const CallbackT& Callback) {
  if (interpret_type == InterpretType::kCompute) {
    phy_instr_operand.ForEachMutMirroredObject([&](MirroredObject* infer, MirroredObject* compute) {
      if (infer != nullptr) { Callback(infer); }
    });
  } else if (interpret_type == InterpretType::kInfer) {
    // Do nothing
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachMutMirroredObject(
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

namespace {

template<typename CallbackT>
void ForEachMutMirroredObject4MutPhyInstrOperand(InterpretType interpret_type,
                                                 const PhyInstrOperand& phy_instr_operand,
                                                 const CallbackT& Callback) {
  if (interpret_type == InterpretType::kCompute) {
    phy_instr_operand.ForEachMutMirroredObject(
        [&](MirroredObject* infer, MirroredObject* compute) { Callback(compute); });
  } else if (interpret_type == InterpretType::kInfer) {
    phy_instr_operand.ForEachMutMirroredObject([&](MirroredObject* infer, MirroredObject* compute) {
      if (infer != nullptr) { Callback(infer); }
    });
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

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

namespace {

template<typename CallbackT>
void ForEachMutMirroredObject4Mut2PhyInstrOperand(InterpretType interpret_type,
                                                  const PhyInstrOperand& phy_instr_operand,
                                                  const CallbackT& Callback) {
  if (interpret_type == InterpretType::kCompute) {
    phy_instr_operand.ForEachMut2MirroredObject(
        [&](MirroredObject* infer, MirroredObject* compute) {
          if (infer != nullptr) { Callback(infer); }
          Callback(compute);
        });
  } else if (interpret_type == InterpretType::kInfer) {
    phy_instr_operand.ForEachMut2MirroredObject(
        [&](MirroredObject* infer, MirroredObject* compute) {
          if (infer != nullptr) { Callback(infer); }
        });
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
void VirtualMachine::ForEachMutMirroredObject(
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

RwMutexedObjectAccess* VirtualMachine::ConsumeMirroredObject(OperandAccessType access_type,
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

void VirtualMachine::ConnectInstruction(Instruction* src_instruction,
                                        Instruction* dst_instruction) {
  CHECK_NE(src_instruction, dst_instruction);
  auto edge = intrusive::make_shared<InstructionEdge>(src_instruction, dst_instruction);
  src_instruction->mut_out_edges()->PushBack(edge.Mutable());
  dst_instruction->mut_in_edges()->PushBack(edge.Mutable());
}

void VirtualMachine::ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                                            NewInstructionList* new_instruction_list) {
  INTRUSIVE_FOR_EACH_PTR(instruction, new_instruction_list) {
    int64_t global_device_id = instruction->stream().global_device_id();
    const InterpretType interpret_type = instruction->stream().stream_type_id().interpret_type();
    auto ConsumeConstMirroredObject = [&](MirroredObject* mirrored_object) {
      ConsumeMirroredObject(kConstOperandAccess, mirrored_object, instruction);
    };
    auto ConsumeMutMirroredObject = [&](MirroredObject* mirrored_object) {
      ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, instruction);
    };
    auto ConsumeDelMirroredObject = [&](MirroredObject* mirrored_object) {
      auto* access = ConsumeMirroredObject(kMutableOperandAccess, mirrored_object, instruction);
      CHECK(!mirrored_object->has_deleting_access());
      mirrored_object->set_deleting_access(access);
    };
    const auto& phy_instr_operand = instruction->instr_msg().phy_instr_operand();
    if (phy_instr_operand) {
      ForEachMutMirroredObject4Mut2PhyInstrOperand(interpret_type, *phy_instr_operand,
                                                   ConsumeMutMirroredObject);
      ForEachMutMirroredObject4MutPhyInstrOperand(interpret_type, *phy_instr_operand,
                                                  ConsumeMutMirroredObject);
    }
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
    if (phy_instr_operand) {
      ForEachConstMirroredObject4MutPhyInstrOperand(interpret_type, *phy_instr_operand,
                                                    ConsumeConstMirroredObject);
      ForEachConstMirroredObject4ConstPhyInstrOperand(interpret_type, *phy_instr_operand,
                                                      ConsumeConstMirroredObject);
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
}

bool VirtualMachine::Dispatchable(Instruction* instruction) const {
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

// Dispatch ready instructions.
// Collect prescheduled instructions onto ready_instruction_list_.
void VirtualMachine::DispatchAndPrescheduleInstructions() {
  if (mut_ready_instruction_list()->size() == 0) { return; }
  ReadyInstructionList tmp_ready_instruction_list;
  mut_ready_instruction_list()->MoveTo(&tmp_ready_instruction_list);
  INTRUSIVE_FOR_EACH(instruction, &tmp_ready_instruction_list) {
    tmp_ready_instruction_list.Erase(instruction.Mutable());
    DispatchInstruction(instruction.Mutable());
    // preschedule instructions
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_out_edges()) {
      TryMoveFromWaitingToReady(edge->mut_dst_instruction());
    }
  }
}

void VirtualMachine::MoveToReadyOrWaiting(NewInstructionList* new_instruction_list) {
  INTRUSIVE_FOR_EACH_PTR(instruction, new_instruction_list) {
    if (Dispatchable(instruction)) {
      mut_ready_instruction_list()->PushBack(instruction);
      new_instruction_list->Erase(instruction);
    }
  }
  new_instruction_list->MoveTo(mut_waiting_instruction_list());
}

void VirtualMachine::DispatchInstruction(Instruction* instruction) {
  OF_PROFILER_RANGE_PUSH("Dispatch-" + instruction->instr_msg().instr_type_name());
  mut_vm_stat_running_instruction_list()->PushBack(instruction);
  auto* stream = instruction->mut_stream();
  stream->mut_running_instruction_list()->PushBack(instruction);
  if (stream->active_stream_hook().empty()) { mut_active_stream_list()->PushBack(stream); }
  const auto& stream_type = stream->stream_type();
  if (stream_type.SharingVirtualMachineThread()) {
    stream_type.Run(this, instruction);
  } else {
    stream->mut_thread_ctx()->mut_pending_instruction_list()->PushBack(instruction);
  }
  OF_PROFILER_RANGE_POP();
}

void VirtualMachine::__Init__(const VmDesc& vm_desc) {
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
        CHECK(stream_rt_desc->mut_stream_id2stream()->Insert(stream.Mutable()).second);
        thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
      }
    }
  }
}

int64_t InstructionMaxRunningSeconds() { return 60 * 5; }

Maybe<void> VirtualMachine::Receive(InstructionMsgList* compute_instr_msg_list) {
  CHECK_OR_RETURN(!pthread_fork::IsForkedSubProcess())
      << "Cannot run OneFlow in forked subprocess. Please add "
         "'multiprocessing.set_start_method(\"spawn\")' in '__main__' if you are using Python's "
         "multiprocessing";
  InstructionMsgList new_instr_msg_list;
  INTRUSIVE_FOR_EACH_PTR(compute_instr_msg, compute_instr_msg_list) {
    if (!compute_instr_msg->phy_instr_operand()) {
      new_instr_msg_list.EmplaceBack(compute_instr_msg->MakeInferInstrMsg());
    }
    compute_instr_msg_list->MoveToDstBack(compute_instr_msg, &new_instr_msg_list);
  }
  const int64_t kHighWaterMark = GetInstructionHighWaterMark();
  const int64_t kLowWaterMark = GetInstructionLowWaterMark();
  if (*mut_flying_instruction_cnt() > kHighWaterMark) {
    JUST(Global<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
      const auto& NeedSpin = [&] { return *mut_flying_instruction_cnt() > kLowWaterMark; };
      while (true) {
        int64_t last_cnt = *mut_flying_instruction_cnt();
        const auto& ret = TRY(SpinWaitUntilTimeout(NeedSpin, InstructionMaxRunningSeconds()));
        if (ret.IsOk()) { break; }
        CHECK_NE_OR_RETURN(last_cnt, *mut_flying_instruction_cnt())
            << Error::UnimplementedError() << "The virtual machine don't respond in "
            << InstructionMaxRunningSeconds() << " seconds.";
      }
      return Maybe<void>::Ok();
    }));
  }
  mut_pending_msg_list()->MoveFrom(&new_instr_msg_list);
  return Maybe<void>::Ok();
}

Maybe<void> VirtualMachine::Receive(intrusive::shared_ptr<InstructionMsg>&& compute_instr_msg) {
  InstructionMsgList instr_msg_list;
  instr_msg_list.EmplaceBack(std::move(compute_instr_msg));
  return Receive(&instr_msg_list);
}

// TODO(lixinqi): refactor to being trigger inside TryReleaseFinishedInstructions
void VirtualMachine::TryRunFrontSeqInstruction() {
  auto* instruction = mut_front_seq_compute_instr_list()->Begin();
  if (instruction == nullptr) { return; }
  const auto& instr_type_id = instruction->instr_msg().instr_type_id();
  const auto& instruction_type = instr_type_id.instruction_type();
  if (!instruction_type.IsFrontSequential()) { return; }
  // All instructions before `instruction` are handled now, it's time to handle `instruction`.
  // TODO(lixinqi): Should replace the `if` with
  // CHECK(instruction->vm_stat_running_instruction_hook().empty()) ?
  if (!instruction->vm_stat_running_instruction_hook().empty()) { return; }
  const StreamType& stream_type = instr_type_id.stream_type_id().stream_type();
  if (stream_type.SharingVirtualMachineThread()) {
    stream_type.Run(this, instruction);
    mut_front_seq_compute_instr_list()->Erase(instruction);
  } else {
    mut_ready_instruction_list()->PushBack(instruction);
  }
}

void VirtualMachine::TryDeleteLogicalObjects() {
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

void VirtualMachine::TryMoveFromWaitingToReady(Instruction* instruction) {
  if (Dispatchable(instruction)) {
    // For memory safety, do not swap the following two lines.
    mut_ready_instruction_list()->PushBack(instruction);
    mut_waiting_instruction_list()->Erase(instruction);
  }
}

void VirtualMachine::Schedule() {
  INTRUSIVE_FOR_EACH_PTR(stream, mut_active_stream_list()) {
    // Collect ready instructions onto ready_instruction_list_.
    TryReleaseFinishedInstructions(stream);
    if (stream->running_instruction_list().empty()) { mut_active_stream_list()->Erase(stream); }
  }
  TryDeleteLogicalObjects();
  // Try run sequential instructions
  TryRunFrontSeqInstruction();
  // Use thread_unsafe_size to avoid acquiring mutex lock.
  // The inconsistency between pending_msg_list.list_head_.list_head_.container_ and
  // pending_msg_list.list_head_.list_head_.size_ is not a fatal error because
  // VirtualMachine::Schedule is always in a buzy loop. All instructions will get handled
  // eventually.
  //  VirtualMachine::Receive may be less effiencient if the thread safe version
  //  `pending_msg_list().size()` used here, because VirtualMachine::Schedule is more likely to get
  //  the mutex lock.
  if (pending_msg_list().thread_unsafe_size() > 0) {
    TmpPendingInstrMsgList tmp_pending_msg_list;
    // MoveTo is under a lock.
    mut_pending_msg_list()->MoveTo(&tmp_pending_msg_list);
    FilterAndRunInstructionsInAdvance(&tmp_pending_msg_list);
    NewInstructionList new_instruction_list;
    MakeInstructions(&tmp_pending_msg_list, /*out*/ &new_instruction_list);
    ConsumeMirroredObjects(mut_id2logical_object(), &new_instruction_list);
    MoveToReadyOrWaiting(&new_instruction_list);
  }
  // Dispatch ready instructions and put prescheduled instructions onto ready_instruction_list_.
  DispatchAndPrescheduleInstructions();
  *mut_flying_instruction_cnt() = mut_waiting_instruction_list()->size()
                                  + mut_ready_instruction_list()->size()
                                  + mut_vm_stat_running_instruction_list()->size();
}

bool VirtualMachine::ThreadUnsafeEmpty() const {
  return pending_msg_list().thread_unsafe_size() == 0 && waiting_instruction_list().empty()
         && active_stream_list().empty() && front_seq_compute_instr_list().empty()
         && flying_instruction_cnt() == 0;
}

bool VirtualMachine::Empty() const {
  return pending_msg_list().empty() && waiting_instruction_list().empty()
         && active_stream_list().empty() && front_seq_compute_instr_list().empty()
         && flying_instruction_cnt() == 0;
}

}  // namespace vm
}  // namespace oneflow
