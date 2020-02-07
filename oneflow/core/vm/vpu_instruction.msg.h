#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_

#include <cstring>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vpu_desc.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vpu_instruction.h"
#include "oneflow/core/vm/vpu_instruction_msg_observer.h"
#include "oneflow/core/vm/vpu_instruction_status_querier.h"

namespace oneflow {

static const int kVpuInstructionOprandLimit = 64;

// clang-format off
BEGIN_FLAT_MSG(VpuInstructionProto);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, vpu_instruction_id);
  FLAT_MSG_DEFINE_OPTIONAL(VpuTypeId, vpu_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(VpuInstructionOpcode, opcode);
  FLAT_MSG_DEFINE_REPEATED(VpuInstructionOprand, operand, kVpuInstructionOprandLimit);
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

// clang-format off
BEGIN_OBJECT_MSG(VpuInstructionCtx);
  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuId, vpu_id);
  OBJECT_MSG_DEFINE_OPTIONAL(VpuInstructionMsg, vpu_instruction_msg);
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuInstruction*, vpu_instruction); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instruction_link);

  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, vpu_instruction_oprand_index,
                                  oprand_index2object_access);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vpu_instruction_oprand_ready_link,
                              ready_oprand_list);
END_OBJECT_MSG(VpuInstructionCtx);
// clang-format on

class OBJECT_MSG_TYPE(VpuCtx);

// clang-format off
BEGIN_OBJECT_MSG(RunningVpuInstructionPackage);
 public:
  template<typename NewQuerierT>
  void __Init__(OBJECT_MSG_TYPE(VpuCtx)* vpu, const NewQuerierT& NewQuerier) {
    set_vpu(vpu);
    mutable_status_querier()->__Init__(NewQuerier);
  }
  bool Done() const { return status_querier()->Done(); }

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(VpuCtx)*, vpu);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VpuInstructionStatusQuerier>, status_querier); 
  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_link, vpu_instruction_list);
  OBJECT_MSG_DEFINE_LIST_LINK(running_vpu_instruction_package_link);
END_OBJECT_MSG(RunningVpuInstructionPackage);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuCtx);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VpuInstructionBuilder*, vpu_instruction_builder); 
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_link);
  OBJECT_MSG_DEFINE_SKIPLIST_FLAT_MSG_KEY(7, VpuId, vpu_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_link, pending_vpu_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(RunningVpuInstructionPackage, running_vpu_instruction_package_link,
                              running_vpu_instruction_package_list);
END_OBJECT_MSG(VpuCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuSetCtx);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VpuTypeId, vpu_type_id);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_set_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuCtx, vpu_link, vpu_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(RunningVpuInstructionPackage, running_vpu_instruction_package_link,
                              launched_vpu_instruction_package_list);
END_OBJECT_MSG(VpuSetCtx);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuSchedulerCtx);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, machine_id);

  //links
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionMsg, vpu_instruction_msg_link,
                              vpu_instruction_msg_pending_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionMsg, vpu_instruction_msg_link,
                              tmp_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VpuCtx, vpu_id, vpu_id2vpu);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuSetCtx, vpu_set_link, vpu_set_list);
  OBJECT_MSG_DEFINE_MAP_HEAD(LogicalObject, logical_object_id, id2logical_object);
END_OBJECT_MSG(VpuSchedulerCtx);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_MSG_H_
