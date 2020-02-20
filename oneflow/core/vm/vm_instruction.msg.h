#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction_msg_observer.h"
#include "oneflow/core/vm/vm_instruction_status_querier.h"

namespace oneflow {

static const int kVmInstructionOperandLimit = 64;

// clang-format off
BEGIN_FLAT_MSG(VmInstructionProto);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, vm_instruction_id);
  FLAT_MSG_DEFINE_OPTIONAL(VmStreamTypeId, vm_stream_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(VmInstructionOpcode, opcode);
  FLAT_MSG_DEFINE_REPEATED(VmInstructionOperand, operand, kVmInstructionOperandLimit);
  FLAT_MSG_DEFINE_OPTIONAL(VmStreamMask, vm_stream_mask);
END_FLAT_MSG(VmInstructionProto);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VmInstructionMsg);
  PUBLIC void __Init__() { __Init__(&VmInstructionMsgNoneObserver::NewObserver); }
  PUBLIC template<typename NewObserverT>
  void __Init__(const NewObserverT& NewObserver) { mutable_observer()->__Init__(NewObserver); }

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VmInstructionProto, vm_instruction_proto);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VmInstructionMsgObserver>, observer);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instruction_msg_link);
END_OBJECT_MSG(VmInstructionMsg);
// clang-format on

class VmStream;

// clang-format off
BEGIN_OBJECT_MSG(VmInstruction);
  // methods
  PUBLIC void __Init__(VmInstructionMsg* vm_instruction_msg, VmStream* vm_stream);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VmInstructionMsg, vm_instruction_msg);
  OBJECT_MSG_DEFINE_RAW_PTR(VmStream, vm_stream); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instruction_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vm_instr_operand_link, waiting_operand_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vm_instr_operand_link, holding_operand_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, logical_object_id_value,
                                  logical_object_id2operand_access);
END_OBJECT_MSG(VmInstruction);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VmInstructionPackage);
  // methods
  PUBLIC void __Init__(VmStream* vm_stream);
  PUBLIC bool Done() const { return status_querier()->Done(); }

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(VmStream, vm_stream);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VmInstructionStatusQuerier>, status_querier); 

  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstruction, vm_instruction_link, vm_instruction_list);
  OBJECT_MSG_DEFINE_LIST_LINK(waiting_pkg_link);
  OBJECT_MSG_DEFINE_LIST_LINK(running_pkg_link);
END_OBJECT_MSG(VmInstructionPackage);
// clang-format on

class VmThread;

// clang-format off
BEGIN_OBJECT_MSG(VmStream);
  // methods
  PUBLIC void __Init__(VmThread* vm_thread, const VmStreamId& vm_stream_id) {
    set_vm_thread(vm_thread);
    mut_vm_stream_id()->CopyFrom(vm_stream_id);
  }
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(VmThread, vm_thread); 
  OBJECT_MSG_DEFINE_FLAT_MSG(VmStreamId, vm_stream_id);
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(active_vm_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(tmp_active_vm_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(vm_thread_vm_stream_link);
  OBJECT_MSG_DEFINE_MAP_KEY(int64_t, parallel_id);
  // collect_vm_instruction_list used by VmScheduler
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstruction, vm_instruction_link,
                              collect_vm_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstructionPackage, running_pkg_link, running_pkg_list);
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(VmInstructionPackage, waiting_pkg_link, waiting_pkg_list);
END_OBJECT_MSG(VmStream);
// clang-format on

// Rt is short for Runtime
// clang-format off
BEGIN_OBJECT_MSG(VmStreamRtDesc);
  // methods
  PUBLIC void __Init__(const VmStreamDesc* vm_stream_desc);

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VmStreamType, vm_stream_type); 
  OBJECT_MSG_DEFINE_RAW_PTR(const VmStreamDesc, vm_stream_desc); 
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, VmStreamTypeId, vm_stream_type_id);
  OBJECT_MSG_DEFINE_MAP_HEAD(VmStream, parallel_id, parallel_id2vm_stream);
END_OBJECT_MSG(VmStreamRtDesc);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VmThread);
  // methods
  PUBLIC void __Init__(const VmStreamRtDesc& vm_stream_rt_desc) {
    set_vm_stream_rt_desc(&vm_stream_rt_desc);
  }
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VmStreamRtDesc, vm_stream_rt_desc); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_thread_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmStream, vm_thread_vm_stream_link, vm_stream_list);
END_OBJECT_MSG(VmThread);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
