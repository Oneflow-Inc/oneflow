#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include <mutex>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vpu_type_desc.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vpu_instruction.h"
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
 public:
  void __Init__() { __Init__(&VpuInstructionMsgNoneObserver::NewObserver); }
  template<typename NewObserverT>
  void __Init__(const NewObserverT& NewObserver) { mutable_observer()->__Init__(NewObserver); }

  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuInstructionProto, vpu_instruction_proto);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VpuInstructionMsgObserver>, observer);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instruction_msg_link);
END_OBJECT_MSG(VpuInstructionMsg);
// clang-format on

class OBJECT_MSG_TYPE(VpuCtx);

// clang-format off
BEGIN_OBJECT_MSG(VpuInstructionCtx);
  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuId, vpu_id);
  OBJECT_MSG_DEFINE_OPTIONAL(VpuInstructionMsg, vpu_instruction_msg);
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuInstruction*, vpu_instruction); 
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(VpuCtx)*, vpu_ctx); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instruction_ctx_link);

  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstrOperandAccess, vpu_instr_operand_link, waiting_operand_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstrOperandAccess, vpu_instr_operand_link, holding_operand_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VpuInstrOperandAccess, logical_object_id,
                                  logical_object_id2operand_access);
END_OBJECT_MSG(VpuInstructionCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(RunningVpuInstructionPackage);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(VpuCtx)*, vpu);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VpuInstructionStatusQuerier>, status_querier); 

  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, vpu_instruction_list);
  OBJECT_MSG_DEFINE_LIST_LINK(running_pkg_link);
  OBJECT_MSG_DEFINE_LIST_LINK(launched_pkg_link);

  // methods
 public:
  template<typename NewQuerierT>
  void __Init__(OBJECT_MSG_TYPE(VpuCtx)* vpu, const NewQuerierT& NewQuerier) {
    set_vpu(vpu);
    mutable_status_querier()->__Init__(NewQuerier);
  }
  bool Done() const { return status_querier()->Done(); }

END_OBJECT_MSG(RunningVpuInstructionPackage);
// clang-format on

class OBJECT_MSG_TYPE(VpuSetCtx);

// clang-format off
BEGIN_OBJECT_MSG(VpuCtx);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuInstructionBuilder*, vpu_instruction_builder); 
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(VpuTypeDesc)*, vpu_type_desc); 
  OBJECT_MSG_DEFINE_RAW_PTR(const OBJECT_MSG_TYPE(VpuSetCtx)*, vpu_set_ctx); 
  // for pending_pkg_list only
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<std::mutex>, pending_list_mutex);  
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_link);
  OBJECT_MSG_DEFINE_SKIPLIST_FLAT_MSG_KEY(7, VpuId, vpu_id);
  // collect_vpu_instruction_list used by VpuScheduler
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link,
                              collect_vpu_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(RunningVpuInstructionPackage, running_pkg_link, pending_pkg_list);

  // methods
 public:
  void __Init__(const VpuInstructionBuilder* builder, OBJECT_MSG_TYPE(VpuSetCtx)* vpu_set_ctx) {
    set_vpu_instruction_builder(builder);
    set_vpu_set_ctx(vpu_set_ctx);
    mutable_pending_list_mutex()->__Init__();
  }
END_OBJECT_MSG(VpuCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuSetCtx);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VpuTypeId, vpu_type_id);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_set_ctx_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuCtx, vpu_link, vpu_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(RunningVpuInstructionPackage, launched_pkg_link, launched_pkg_list);
END_OBJECT_MSG(VpuSetCtx);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
