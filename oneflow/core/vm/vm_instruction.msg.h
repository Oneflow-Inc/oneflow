#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vm_stream_type.h"

namespace oneflow {

// clang-format off
FLAT_MSG_BEGIN(VmInstructionId);
  FLAT_MSG_DEFINE_OPTIONAL(VmStreamTypeId, vm_stream_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(VmInstructionOpcode, opcode);
  FLAT_MSG_DEFINE_OPTIONAL(bool, is_local);
FLAT_MSG_END(VmInstructionId);
// clang-format on

class MirroredObject;

// clang-format off
OBJECT_MSG_BEGIN(VmInstructionMsg);
  // methods
  PUBLIC VmInstructionOperand* add_operand();

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VmInstructionId, vm_instr_id);
  OBJECT_MSG_DEFINE_STRUCT(std::vector<FlatMsg<VmInstructionOperand>>, operand);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instr_msg_link);
OBJECT_MSG_END(VmInstructionMsg);
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(VmInstruction);
  // methods
  PUBLIC void __Init__(VmInstrChain* vm_instr_chain, VmInstructionMsg* vm_instr_msg) {
    set_vm_instr_chain(vm_instr_chain);
    reset_vm_instr_msg(vm_instr_msg);
  }
  PUBLIC MirroredObject* FindMirroredObjectByOperand(const MirroredObjectOperand& operand,
                                                     int64_t default_parallel_id);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VmInstructionMsg, vm_instr_msg);
  OBJECT_MSG_DEFINE_PTR(VmInstrChain, vm_instr_chain);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instruction_link);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, mirrored_object_id, mirrored_object_id2access);
OBJECT_MSG_END(VmInstruction);
// clang-format on

static const int kVmInstructionStatusBufferBytes = 32;
// clang-format off
FLAT_MSG_BEGIN(VmInstructionStatusBuffer);
  FLAT_MSG_DEFINE_REPEATED(char, buffer, kVmInstructionStatusBufferBytes);
FLAT_MSG_END(VmInstructionStatusBuffer);
// clang-format on

class VmInstrChain;
// clang-format off
OBJECT_MSG_BEGIN(VmInstrChainEdge);
  // methods
  PUBLIC void __Init__(VmInstrChain* src_vm_instr_chain, VmInstrChain* dst_vm_instr_chain) {
    set_src_vm_instr_chain(src_vm_instr_chain);
    set_dst_vm_instr_chain(dst_vm_instr_chain);
  }
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, VmInstrChain*, src_vm_instr_chain);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, VmInstrChain*, dst_vm_instr_chain);
OBJECT_MSG_END(VmInstrChainEdge);
// clang-format on

class VmStream;
// clang-format off
OBJECT_MSG_BEGIN(VmInstrChain);
  // methods
  PUBLIC void __Init__(VmInstructionMsg* vm_instr_msg, VmStream* vm_stream);
  PUBLIC void __Delete__();
  PUBLIC bool Done() const;

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VmInstructionStatusBuffer, status_buffer);
  OBJECT_MSG_DEFINE_PTR(VmStream, vm_stream); 
  OBJECT_MSG_DEFINE_PTR(const VmStreamType, vm_stream_type);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instr_chain_link);
  OBJECT_MSG_DEFINE_LIST_LINK(pending_chain_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstruction, vm_instruction_link, vm_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VmInstrChainEdge, src_vm_instr_chain, in_edges);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VmInstrChainEdge, dst_vm_instr_chain, out_edges);
OBJECT_MSG_END(VmInstrChain);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
