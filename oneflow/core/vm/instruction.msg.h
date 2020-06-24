#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/vm_object.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/instruction.pb.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(InstructionOperandList);
  OBJECT_MSG_DEFINE_STRUCT(std::vector<FlatMsg<InstructionOperand>>, operand);
OBJECT_MSG_END(InstructionOperandList);

OBJECT_MSG_BEGIN(InstructionMsg);
  // methods
  PUBLIC void __Init__() { mutable_operand_list(); }
  PUBLIC void __Init__(const std::string& instr_type_name);
  PUBLIC void __Init__(const InstructionProto& proto);
  PUBLIC void __Init__(const InstructionMsg& instr_msg);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_parallel_desc(int64_t symbol_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_double_operand(double double_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_int64_operand(int64_t int64_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_uint64_operand(uint64_t uint64_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_bool_operand(bool bool_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_separator();
  PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(ObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(ObjectId logical_object_id, const SoleMirroredObject&);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(ObjectId logical_object_id, const AllMirroredObject&);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_symbol_operand(ObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(ObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(ObjectId logical_object_id, const SoleMirroredObject&);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(ObjectId logical_object_id, const AllMirroredObject&);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_init_symbol_operand(ObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id, const SoleMirroredObject&);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(ObjectId logical_object_id, const AllMirroredObject&);
  PUBLIC const std::vector<FlatMsg<InstructionOperand>>& operand() const {
    return operand_list().operand();
  }
  PUBLIC std::vector<FlatMsg<InstructionOperand>>* mut_operand() {
    return mut_operand_list()->mut_operand();
  }
  PUBLIC std::vector<FlatMsg<InstructionOperand>>* mutable_operand() {
    return mutable_operand_list()->mut_operand();
  }
  PUBLIC ObjectMsgPtr<InstructionMsg> MakeInferInstrMsg() const;

  // fields
  OBJECT_MSG_DEFINE_STRUCT(InstrTypeId, instr_type_id);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, parallel_desc_symbol_id);
  OBJECT_MSG_DEFINE_OPTIONAL(InstructionOperandList, operand_list);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(instr_msg_link);

  // private methods
  PRIVATE InstructionOperand* add_instr_operand();
OBJECT_MSG_END(InstructionMsg);
// clang-format on

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
  PUBLIC void __Init__(Instruction* src_instruction, Instruction* dst_instruction) {
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
  PUBLIC void __Init__(InstructionMsg* instr_msg, Stream* stream, const std::shared_ptr<ParallelDesc>& parallel_desc);
  PUBLIC void __Delete__();
  PUBLIC bool Done() const;
  PUBLIC const StreamType& stream_type() const;

  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      const RwMutexedObject* operand_type(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_type(operand, GetOperandDefaultGlobalDeviceId());
  }
  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      const RwMutexedObject* operand_value(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_value(operand, GetOperandDefaultGlobalDeviceId());
  }
  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      RwMutexedObject* mut_operand_type(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_type(operand, GetOperandDefaultGlobalDeviceId());
  }
  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      RwMutexedObject* mut_operand_value(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_value(operand, GetOperandDefaultGlobalDeviceId());
  }

  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_type<mem_zone_modifier>(operand.operand());
  }
  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const RwMutexedObject* operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_value<mem_zone_modifier>(operand.operand());
  }
  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_type<mem_zone_modifier>(operand.operand());
  }
  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  RwMutexedObject* mut_operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_value<mem_zone_modifier>(operand.operand());
  }

  PUBLIC template<InterpretType interpret_type>
         MirroredObject* MutMirroredObject(const MutOperand& mut_operand) {
    return MirroredObjectUtil<interpret_type>::Mut(this, mut_operand);
  }

  PUBLIC template<InterpretType interpret_type>
         const MirroredObject* GetMirroredObject(const ConstOperand& const_operand) const {
    return MirroredObjectUtil<interpret_type>::Get(*this, const_operand);
  }

  PUBLIC MirroredObject* mut_type_mirrored_object(const MutOperand& mut_operand);
  PUBLIC MirroredObject* mut_value_mirrored_object(const MutOperand& mut_operand);

  // private methods
  PRIVATE template<int64_t(*TransformLogicalObjectId)(int64_t)>
          MirroredObject* MutMirroredObject(const Operand& operand,
                                            int64_t default_global_device_id);
  PRIVATE template<int64_t(*TransformLogicalObjectId)(int64_t)>
          const MirroredObject* GetMirroredObject(const Operand& operand,
                                                  int64_t default_global_device_id) const;
  PRIVATE const RwMutexedObject* operand_type(const Operand& operand,
                                              int64_t default_global_device_id) const;
  PRIVATE const RwMutexedObject* operand_value(const Operand& operand,
                                               int64_t default_global_device_id) const;
  PRIVATE RwMutexedObject* mut_operand_type(const Operand& operand,
                                            int64_t default_global_device_id);
  PRIVATE RwMutexedObject* mut_operand_value(const Operand& operand,  
                                             int64_t default_global_device_id);

  PRIVATE MirroredObject* MutMirroredObject(const Operand& operand,
                                                     int64_t default_global_device_id) {
    return MutMirroredObject<&IdUtil::GetValueId>(operand, default_global_device_id);
  }

  PRIVATE int64_t GetOperandDefaultGlobalDeviceId() const;

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
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstructionEdge, src_instruction, in_edges);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstructionEdge, dst_instruction, out_edges);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(RwMutexedObjectAccess, mirrored_object_id, mirrored_object_id2access);
OBJECT_MSG_END(Instruction);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
