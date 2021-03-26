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
#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/vm_object.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/instruction.cfg.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(InstructionOperandList);
  OBJECT_MSG_DEFINE_STRUCT(std::vector<FlatMsg<InstructionOperand>>, operand);
OBJECT_MSG_END(InstructionOperandList);

OBJECT_MSG_BEGIN(InstructionMsg);
  // methods
  OF_PUBLIC void __Init__();
  OF_PUBLIC void __Init__(const std::string& instr_type_name);
  OF_PUBLIC void __Init__(const InstructionProto& proto);
  OF_PUBLIC void __Init__(const cfg::InstructionProto& proto); 
  OF_PUBLIC void __Init__(const InstructionMsg& instr_msg);

  OF_PUBLIC void ToProto(InstructionProto* proto) const;
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_parallel_desc(int64_t symbol_id);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_double_operand(double double_operand);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_int64_operand(int64_t int64_operand);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_uint64_operand(uint64_t uint64_operand);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_bool_operand(bool bool_operand);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_separator();
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(ObjectId logical_object_id);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(ObjectId logical_object_id, const SoleMirroredObject&);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(ObjectId logical_object_id, const AllMirroredObject&);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_symbol_operand(ObjectId logical_object_id);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(ObjectId logical_object_id);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(ObjectId logical_object_id, const SoleMirroredObject&);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(ObjectId logical_object_id, const AllMirroredObject&);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_init_symbol_operand(ObjectId logical_object_id);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id, const SoleMirroredObject&);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id, const AllMirroredObject&);
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> add_del_object_operand(ObjectId logical_object_id);
  OF_PUBLIC const std::vector<FlatMsg<InstructionOperand>>& operand() const {
    return operand_list().operand();
  }
  OF_PUBLIC std::vector<FlatMsg<InstructionOperand>>* mut_operand() {
    return mut_operand_list()->mut_operand();
  }
  OF_PUBLIC std::vector<FlatMsg<InstructionOperand>>* mutable_operand() {
    return mutable_operand_list()->mut_operand();
  }
  OF_PUBLIC ObjectMsgPtr<InstructionMsg> MakeInferInstrMsg() const;

  // fields
  OBJECT_MSG_DEFINE_STRUCT(InstrTypeId, instr_type_id);
  // instr_type_name is a necessary reduandant field for method ToProto
  OBJECT_MSG_DEFINE_STRUCT(std::string, instr_type_name);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, parallel_desc_symbol_id);
  OBJECT_MSG_DEFINE_OPTIONAL(InstructionOperandList, operand_list);
  OBJECT_MSG_DEFINE_STRUCT(std::shared_ptr<std::function<void()>>, no_arg_callback);


  // links
  OBJECT_MSG_DEFINE_LIST_LINK(instr_msg_link);

  // private methods
  OF_PRIVATE InstructionOperand* add_instr_operand();
OBJECT_MSG_END(InstructionMsg);
// clang-format on

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

template<OperandMemZoneModifier mem_zone_modifier>
void CheckOperand(const Operand& operand);

static const int kInstructionStatusBufferBytes = 32;
// clang-format off
FLAT_MSG_BEGIN(InstructionStatusBuffer);
  FLAT_MSG_DEFINE_REPEATED(char, buffer, kInstructionStatusBufferBytes);
FLAT_MSG_END(InstructionStatusBuffer);
// clang-format on

class Instruction;
// clang-format off
OBJECT_MSG_BEGIN(InstructionEdge);
  // methods
  OF_PUBLIC void __Init__(Instruction* src_instruction, Instruction* dst_instruction) {
    set_src_instruction(src_instruction);
    set_dst_instruction(dst_instruction);
  }
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, Instruction*, src_instruction);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, Instruction*, dst_instruction);
OBJECT_MSG_END(InstructionEdge);
// clang-format on

class Stream;
// clang-format off
OBJECT_MSG_BEGIN(Instruction);
  // methods
  OF_PUBLIC void __Init__(InstructionMsg* instr_msg, Stream* stream, const std::shared_ptr<ParallelDesc>& parallel_desc);
  OF_PUBLIC void __Delete__();
  OF_PUBLIC bool Done() const;
  OF_PUBLIC const StreamType& stream_type() const;

  OF_PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      const RwMutexedObject* operand_type(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_type(operand, GetOperandDefaultGlobalDeviceId());
  }
  OF_PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      const RwMutexedObject* operand_value(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_value(operand, GetOperandDefaultGlobalDeviceId());
  }
  OF_PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      RwMutexedObject* mut_operand_type(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_type(operand, GetOperandDefaultGlobalDeviceId());
  }
  OF_PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      RwMutexedObject* mut_operand_value(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_value(operand, GetOperandDefaultGlobalDeviceId());
  }

  OF_PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_type<mem_zone_modifier>(operand.operand());
  }
  OF_PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_value<mem_zone_modifier>(operand.operand());
  }
  OF_PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_type<mem_zone_modifier>(operand.operand());
  }
  OF_PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_value<mem_zone_modifier>(operand.operand());
  }

  OF_PUBLIC template<InterpretType interpret_type>
         MirroredObject* MutMirroredObject(const MutOperand& mut_operand) {
    return MirroredObjectUtil<interpret_type>::Mut(this, mut_operand);
  }

  OF_PUBLIC template<InterpretType interpret_type>
         const MirroredObject* GetMirroredObject(const ConstOperand& const_operand) const {
    return MirroredObjectUtil<interpret_type>::Get(*this, const_operand);
  }

  OF_PUBLIC MirroredObject* mut_type_mirrored_object(const MutOperand& mut_operand);
  OF_PUBLIC MirroredObject* mut_value_mirrored_object(const MutOperand& mut_operand);

  // private methods
  OF_PRIVATE template<int64_t(*TransformLogicalObjectId)(int64_t)>
          MirroredObject* MutMirroredObject(const Operand& operand,
                                            int64_t default_global_device_id);
  OF_PRIVATE template<int64_t(*TransformLogicalObjectId)(int64_t)>
          const MirroredObject* GetMirroredObject(const Operand& operand,
                                                  int64_t default_global_device_id) const;
  OF_PRIVATE const RwMutexedObject* operand_type(const Operand& operand,
                                              int64_t default_global_device_id) const;
  OF_PRIVATE const RwMutexedObject* operand_value(const Operand& operand,
                                               int64_t default_global_device_id) const;
  OF_PRIVATE RwMutexedObject* mut_operand_type(const Operand& operand,
                                            int64_t default_global_device_id);
  OF_PRIVATE RwMutexedObject* mut_operand_value(const Operand& operand,  
                                             int64_t default_global_device_id);

  OF_PRIVATE MirroredObject* MutMirroredObject(const Operand& operand,
                                                     int64_t default_global_device_id) {
    return MutMirroredObject<&IdUtil::GetValueId>(operand, default_global_device_id);
  }

  OF_PRIVATE int64_t GetOperandDefaultGlobalDeviceId() const;

  template<InterpretType interpret_type>
  struct MirroredObjectUtil {
    static const MirroredObject* Get(const Instruction&, const ConstOperand&);
    static MirroredObject* Mut(Instruction*, const MutOperand&);
  };

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(InstructionStatusBuffer, status_buffer);
  OBJECT_MSG_DEFINE_OPTIONAL(InstructionMsg, instr_msg);
  OBJECT_MSG_DEFINE_STRUCT(std::shared_ptr<ParallelDesc>, parallel_desc);
  OBJECT_MSG_DEFINE_PTR(Stream, stream); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(instruction_link);
  OBJECT_MSG_DEFINE_LIST_LINK(pending_instruction_link);
  OBJECT_MSG_DEFINE_LIST_LINK(front_seq_infer_instr_link);
  OBJECT_MSG_DEFINE_LIST_LINK(front_seq_compute_instr_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstructionEdge, src_instruction, in_edges);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstructionEdge, dst_instruction, out_edges);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(RwMutexedObjectAccess, mirrored_object_id, mirrored_object_id2access);
OBJECT_MSG_END(Instruction);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
