#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/logical_object_id.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/instruction.pb.h"

namespace oneflow {
namespace vm {

class MirroredObject;

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
  PUBLIC ObjectMsgPtr<InstructionMsg> add_double_operand(double double_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_int64_operand(int64_t int64_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_uint64_operand(uint64_t uint64_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_bool_operand(bool bool_operand);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_separator();
  PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(LogicalObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(LogicalObjectId logical_object_id, int64_t parallel_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_const_operand(LogicalObjectId logical_object_id, const AllParallelId&);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_const_host_operand(LogicalObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(LogicalObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(LogicalObjectId logical_object_id, int64_t parallel_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut_operand(LogicalObjectId logical_object_id, const AllParallelId&);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_init_const_host_operand(LogicalObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(LogicalObjectId logical_object_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(LogicalObjectId logical_object_id, int64_t parallel_id);
  PUBLIC ObjectMsgPtr<InstructionMsg> add_mut2_operand(LogicalObjectId logical_object_id, const AllParallelId&);
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
  OBJECT_MSG_DEFINE_OPTIONAL(InstructionOperandList, operand_list);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(instr_msg_link);

  // private methods
  PRIVATE InstructionOperand* add_instr_operand();
OBJECT_MSG_END(InstructionMsg);
// clang-format on

template<OperandMemZoneModifier mem_zone_modifier>
void CheckOperand(const Operand& operand);

// clang-format off
OBJECT_MSG_BEGIN(InstrCtx);
  // methods
  PUBLIC void __Init__(InstrChain* instr_chain, InstructionMsg* instr_msg) {
    set_instr_chain(instr_chain);
    reset_instr_msg(instr_msg);
  }
  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      const MirroredObject& operand_type(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_type(operand, GetOperandDefaultParallelId());
  }
  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      const MirroredObject& operand_value(const Operand& operand) const {
    CheckOperand<mem_zone_modifier>(operand);
    return operand_value(operand, GetOperandDefaultParallelId());
  }
  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      MirroredObject* mut_operand_type(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_type(operand, GetOperandDefaultParallelId());
  }
  PUBLIC template<OperandMemZoneModifier mem_zone_modifier>
      MirroredObject* mut_operand_value(const Operand& operand) {
    CheckOperand<mem_zone_modifier>(operand);
    return mut_operand_value(operand, GetOperandDefaultParallelId());
  }

  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const MirroredObject& operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_type<mem_zone_modifier>(operand.operand());
  }
  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  const MirroredObject& operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) const {
    return operand_value<mem_zone_modifier>(operand.operand());
  }
  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  MirroredObject* mut_operand_type(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_type<mem_zone_modifier>(operand.operand());
  }
  PUBLIC template<OperandAccessModifier access_modifier, OperandMemZoneModifier mem_zone_modifier>
  MirroredObject* mut_operand_value(
      const ModifiedOperand<access_modifier, mem_zone_modifier>& operand) {
    return mut_operand_value<mem_zone_modifier>(operand.operand());
  }

  PUBLIC MirroredObject* FindMirroredObjectByOperand(const Operand& operand,
                                                     int64_t default_parallel_id) {
    return FindMirroredObjectByOperand<&GetSelfLogicalObjectId>(operand, default_parallel_id);
  }
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(InstructionMsg, instr_msg);
  OBJECT_MSG_DEFINE_PTR(InstrChain, instr_chain);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(instr_ctx_link);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, mirrored_object_id, mirrored_object_id2access);
  // private methods
  PRIVATE template<int64_t(*TransformLogicalObjectId)(int64_t)>
          MirroredObject* FindMirroredObjectByOperand(const Operand& operand,
                                                      int64_t default_parallel_id);
  PRIVATE template<int64_t(*TransformLogicalObjectId)(int64_t)>
          const MirroredObject* FindMirroredObjectByOperand(const Operand& operand,
                                                            int64_t default_parallel_id) const;
  PRIVATE const MirroredObject& operand_type(const Operand& operand, int64_t default_parallel_id) const;
  PRIVATE const MirroredObject& operand_value(const Operand& operand, int64_t default_parallel_id) const;
  PRIVATE MirroredObject* mut_operand_type(const Operand& operand, int64_t default_parallel_id);
  PRIVATE MirroredObject* mut_operand_value(const Operand& operand, int64_t default_parallel_id);
  PRIVATE int64_t GetOperandDefaultParallelId() const;

OBJECT_MSG_END(InstrCtx);
// clang-format on

static const int kInstructionStatusBufferBytes = 32;
// clang-format off
FLAT_MSG_BEGIN(InstructionStatusBuffer);
  FLAT_MSG_DEFINE_REPEATED(char, buffer, kInstructionStatusBufferBytes);
FLAT_MSG_END(InstructionStatusBuffer);
// clang-format on

class InstrChain;
// clang-format off
OBJECT_MSG_BEGIN(InstrChainEdge);
  // methods
  PUBLIC void __Init__(InstrChain* src_instr_chain, InstrChain* dst_instr_chain) {
    set_src_instr_chain(src_instr_chain);
    set_dst_instr_chain(dst_instr_chain);
  }
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, InstrChain*, src_instr_chain);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, InstrChain*, dst_instr_chain);
OBJECT_MSG_END(InstrChainEdge);
// clang-format on

class Stream;
// clang-format off
OBJECT_MSG_BEGIN(InstrChain);
  // methods
  PUBLIC void __Init__(InstructionMsg* instr_msg, Stream* stream);
  PUBLIC void __Delete__();
  PUBLIC bool Done() const;
  PUBLIC const StreamType& stream_type() const;

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(InstructionStatusBuffer, status_buffer);
  OBJECT_MSG_DEFINE_PTR(Stream, stream); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(instr_chain_link);
  OBJECT_MSG_DEFINE_LIST_LINK(pending_chain_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(InstrCtx, instr_ctx_link, instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstrChainEdge, src_instr_chain, in_edges);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstrChainEdge, dst_instr_chain, out_edges);
OBJECT_MSG_END(InstrChain);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
