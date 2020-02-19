#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vpu_type_desc.msg.h"
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
  FLAT_MSG_DEFINE_OPTIONAL(VpuMask, vpu_mask);
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
BEGIN_OBJECT_MSG(VmInstructionCtx);
  // methods
  PUBLIC void __Init__(VmInstructionMsg* vm_instruction_msg, VmStream* vm_stram);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VmInstructionMsg, vm_instruction_msg);
  OBJECT_MSG_DEFINE_RAW_PTR(const VmInstruction, vm_instruction); 
  OBJECT_MSG_DEFINE_RAW_PTR(VmStream, vm_stram); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instruction_ctx_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vm_instr_operand_link, waiting_operand_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vm_instr_operand_link, holding_operand_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, logical_object_id_value,
                                  logical_object_id2operand_access);
END_OBJECT_MSG(VmInstructionCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(RunningVmInstructionPackage);
  // methods
  PUBLIC void __Init__(VmStream* vm_stram);
  PUBLIC bool Done() const { return status_querier()->Done(); }

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(VmStream, vm_stram);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VmInstructionStatusQuerier>, status_querier); 

  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstructionCtx, vm_instruction_ctx_link, vm_instruction_ctx_list);
  OBJECT_MSG_DEFINE_LIST_LINK(running_pkg_link);
  OBJECT_MSG_DEFINE_LIST_LINK(launched_pkg_link);
END_OBJECT_MSG(RunningVmInstructionPackage);
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
  OBJECT_MSG_DEFINE_LIST_LINK(active_vm_stram_link);
  OBJECT_MSG_DEFINE_LIST_LINK(vm_stram_link_of_vpu_set);
  OBJECT_MSG_DEFINE_LIST_LINK(vm_stram_link_of_vpu_type);
  OBJECT_MSG_DEFINE_MAP_KEY(int64_t, parallel_id);
  // collect_vm_instruction_list used by VpuScheduler
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstructionCtx, vm_instruction_ctx_link,
                              collect_vm_instruction_list);
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(RunningVmInstructionPackage, running_pkg_link,
                                        waiting_pkg_list);
END_OBJECT_MSG(VmStream);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuTypeCtx);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VmStreamType, vm_stream_type); 
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuTypeDesc, vpu_type_desc); 
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, VmStreamTypeId, vm_stream_type_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmStream, vm_stram_link_of_vpu_type, vm_stram_list);
END_OBJECT_MSG(VpuTypeCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VmThread);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuTypeCtx, vpu_type_ctx); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_thread_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmStream, vm_stram_link_of_vpu_set, vm_stram_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(RunningVmInstructionPackage, launched_pkg_link, launched_pkg_list);
END_OBJECT_MSG(VmThread);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
