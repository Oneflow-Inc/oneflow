#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vpu_type_desc.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vpu.h"
#include "oneflow/core/vm/vpu_instruction_msg_observer.h"
#include "oneflow/core/vm/vpu_instruction_status_querier.h"

namespace oneflow {

static const int kVpuInstructionOperandLimit = 64;

// clang-format off
BEGIN_FLAT_MSG(VpuInstructionProto);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, vpu_instruction_id);
  FLAT_MSG_DEFINE_OPTIONAL(VpuTypeId, vpu_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(VpuInstructionOpcode, opcode);
  FLAT_MSG_DEFINE_REPEATED(VpuInstructionOperand, operand, kVpuInstructionOperandLimit);
  FLAT_MSG_DEFINE_OPTIONAL(VpuMask, vpu_mask);
END_FLAT_MSG(VpuInstructionProto);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuInstructionMsg);
  PUBLIC void __Init__() { __Init__(&VpuInstructionMsgNoneObserver::NewObserver); }
  PUBLIC template<typename NewObserverT>
  void __Init__(const NewObserverT& NewObserver) { mutable_observer()->__Init__(NewObserver); }

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuInstructionProto, vpu_instruction_proto);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VpuInstructionMsgObserver>, observer);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instruction_msg_link);
END_OBJECT_MSG(VpuInstructionMsg);
// clang-format on

class VpuCtx;

// clang-format off
BEGIN_OBJECT_MSG(VpuInstructionCtx);
  // methods
  PUBLIC void __Init__(VpuInstructionMsg* vpu_instruction_msg, VpuCtx* vpu_ctx);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VpuInstructionMsg, vpu_instruction_msg);
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuInstruction, vpu_instruction); 
  OBJECT_MSG_DEFINE_RAW_PTR(VpuCtx, vpu_ctx); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instruction_ctx_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vpu_instr_operand_link, waiting_operand_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vpu_instr_operand_link, holding_operand_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, logical_object_id_value,
                                  logical_object_id2operand_access);
END_OBJECT_MSG(VpuInstructionCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(RunningVpuInstructionPackage);
  // methods
  PUBLIC void __Init__(VpuCtx* vpu_ctx);
  PUBLIC bool Done() const { return status_querier()->Done(); }

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(VpuCtx, vpu_ctx);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VpuInstructionStatusQuerier>, status_querier); 

  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, vpu_instruction_ctx_list);
  OBJECT_MSG_DEFINE_LIST_LINK(running_pkg_link);
  OBJECT_MSG_DEFINE_LIST_LINK(launched_pkg_link);
END_OBJECT_MSG(RunningVpuInstructionPackage);
// clang-format on

class VpuSetCtx;

// clang-format off
BEGIN_OBJECT_MSG(VpuCtx);
  // methods
  PUBLIC void __Init__(VpuSetCtx* vpu_set_ctx, const VpuId& vpu_id) {
    set_vpu_set_ctx(vpu_set_ctx);
    mut_vpu_id()->CopyFrom(vpu_id);
  }
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(VpuSetCtx, vpu_set_ctx); 
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuId, vpu_id);
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(active_vpu_ctx_link);
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_ctx_link_of_vpu_set);
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_ctx_link_of_vpu_type);
  OBJECT_MSG_DEFINE_MAP_KEY(int64_t, parallel_id);
  // collect_vpu_instruction_list used by VpuScheduler
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link,
                              collect_vpu_instruction_list);
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(RunningVpuInstructionPackage, running_pkg_link,
                                        waiting_pkg_list);
END_OBJECT_MSG(VpuCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuTypeCtx);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const Vpu, vpu); 
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuTypeDesc, vpu_type_desc); 
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, VpuTypeId, vpu_type_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuCtx, vpu_ctx_link_of_vpu_type, vpu_ctx_list);
END_OBJECT_MSG(VpuTypeCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuSetCtx);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuTypeCtx, vpu_type_ctx); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_set_ctx_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuCtx, vpu_ctx_link_of_vpu_set, vpu_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(RunningVpuInstructionPackage, launched_pkg_link, launched_pkg_list);
END_OBJECT_MSG(VpuSetCtx);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
