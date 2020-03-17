#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction_id.msg.h"
#include "oneflow/core/vm/instruction.pb.h"

namespace oneflow {
namespace vm {

class MirroredObject;

// clang-format off
OBJECT_MSG_BEGIN(InstructionMsg);
  // methods
  PUBLIC void __Init__() {}
  PUBLIC void __Init__(const InstructionProto& proto);
  PUBLIC InstructionOperand* add_operand();

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(InstructionId, vm_instr_id);
  OBJECT_MSG_DEFINE_STRUCT(std::vector<FlatMsg<InstructionOperand>>, operand);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instr_msg_link);
OBJECT_MSG_END(InstructionMsg);
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(Instruction);
  // methods
  PUBLIC void __Init__(InstrChain* vm_instr_chain, InstructionMsg* vm_instr_msg) {
    set_vm_instr_chain(vm_instr_chain);
    reset_vm_instr_msg(vm_instr_msg);
  }
  PUBLIC MirroredObject* FindMirroredObjectByOperand(const MirroredObjectOperand& operand,
                                                     int64_t default_parallel_id);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(InstructionMsg, vm_instr_msg);
  OBJECT_MSG_DEFINE_PTR(InstrChain, vm_instr_chain);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instruction_link);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, mirrored_object_id, mirrored_object_id2access);
OBJECT_MSG_END(Instruction);
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
  PUBLIC void __Init__(InstrChain* src_vm_instr_chain, InstrChain* dst_vm_instr_chain) {
    set_src_vm_instr_chain(src_vm_instr_chain);
    set_dst_vm_instr_chain(dst_vm_instr_chain);
  }
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, InstrChain*, src_vm_instr_chain);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, InstrChain*, dst_vm_instr_chain);
OBJECT_MSG_END(InstrChainEdge);
// clang-format on

class Stream;
// clang-format off
OBJECT_MSG_BEGIN(InstrChain);
  // methods
  PUBLIC void __Init__(InstructionMsg* vm_instr_msg, Stream* vm_stream);
  PUBLIC void __Delete__();
  PUBLIC bool Done() const;

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(InstructionStatusBuffer, status_buffer);
  OBJECT_MSG_DEFINE_PTR(Stream, vm_stream); 
  OBJECT_MSG_DEFINE_PTR(const StreamType, vm_stream_type);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instr_chain_link);
  OBJECT_MSG_DEFINE_LIST_LINK(pending_chain_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, vm_instruction_link, vm_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstrChainEdge, src_vm_instr_chain, in_edges);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(InstrChainEdge, dst_vm_instr_chain, out_edges);
OBJECT_MSG_END(InstrChain);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
