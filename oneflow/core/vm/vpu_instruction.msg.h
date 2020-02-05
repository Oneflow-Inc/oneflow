#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_

#include <cstring>
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"
#include "oneflow/core/vm/vpu_instruction_msg_observer.h"
#include "oneflow/core/vm/vpu_instruction_status_querier.h"

namespace oneflow {

// clang-format off
BEGIN_FLAT_MSG(AllVpuEnabledMask);
END_FLAT_MSG(AllVpuEnabledMask);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuMask);
  FLAT_MSG_DEFINE_ONEOF(mask_type,
      FLAT_MSG_ONEOF_FIELD(AllVpuEnabledMask, all_vpu_enabled)
      FLAT_MSG_ONEOF_FIELD(LogicalObjectId, enabled_parallel_desc_symbol));
END_FLAT_MSG(VpuMask);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(LocalVpu);
END_FLAT_MSG(LocalVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(ControlVpu);
END_FLAT_MSG(ControlVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(HostMemoryVpu);
END_FLAT_MSG(HostMemoryVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(DeviceMemoryVpu);
END_FLAT_MSG(DeviceMemoryVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(CpuDeviceVpu);
END_FLAT_MSG(CpuDeviceVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(GpuDeviceVpu);
END_FLAT_MSG(GpuDeviceVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(H2DTransportVpu);
END_FLAT_MSG(H2DTransportVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(D2HTransportVpu);
END_FLAT_MSG(D2HTransportVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(L2RTransportVpu);
END_FLAT_MSG(L2RTransportVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(R2LTransportVpu);
END_FLAT_MSG(R2LTransportVpu);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuInstructionDesc);
  FLAT_MSG_DEFINE_OPTIONAL(VpuMask, vpu_mask);
  FLAT_MSG_DEFINE_ONEOF(vpu_type,
    FLAT_MSG_ONEOF_FIELD(LocalVpu, local)
    FLAT_MSG_ONEOF_FIELD(ControlVpu, control)
    FLAT_MSG_ONEOF_FIELD(HostMemoryVpu, host_memory)
    FLAT_MSG_ONEOF_FIELD(DeviceMemoryVpu, device_memory)
    FLAT_MSG_ONEOF_FIELD(CpuDeviceVpu, cpu_device)
    FLAT_MSG_ONEOF_FIELD(GpuDeviceVpu, gpu_device)
    FLAT_MSG_ONEOF_FIELD(H2DTransportVpu, h2d_transport)
    FLAT_MSG_ONEOF_FIELD(D2HTransportVpu, d2h_transport)
    FLAT_MSG_ONEOF_FIELD(L2RTransportVpu, l2r_transport)
    FLAT_MSG_ONEOF_FIELD(R2LTransportVpu, r2l_transport));
END_FLAT_MSG(VpuInstructionDesc);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuId);
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(VpuInstructionDesc::_FLAT_MSG_ONEOF_ENUM_TYPE(vpu_type), vpu_type);
  FLAT_MSG_DEFINE_OPTIONAL(int16_t, machine_id);
  FLAT_MSG_DEFINE_OPTIONAL(int16_t, device_id);

  // methods
  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
END_FLAT_MSG(VpuId);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuInstructionOprand);
  FLAT_MSG_DEFINE_ONEOF(oprand_type,
    FLAT_MSG_ONEOF_FIELD(LogicalObjectId, const_oprand)
    FLAT_MSG_ONEOF_FIELD(LogicalObjectId, mutable_oprand));
END_FLAT_MSG(VpuInstructionOprand);
// clang-format on

static const int kVpuInstructionOprandLimit = 64;

// clang-format off
BEGIN_FLAT_MSG(VpuInstructionProto);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, vpu_instruction_id);
  FLAT_MSG_DEFINE_OPTIONAL(VpuInstructionDesc, vpu_instruction_desc);
  FLAT_MSG_DEFINE_REPEATED(VpuInstructionOprand, operand, kVpuInstructionOprandLimit);
END_FLAT_MSG(VpuInstructionProto);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuInstructionMsg);
 public:
  template<typename NewObserverT>
  void __Init__(const NewObserverT& NewObserver) { mutable_observer()->__Init__(NewObserver); }
  void __Init__() { __Init__(&VpuInstructionMsgNoneObserver::NewObserver); }
  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuInstructionProto, vpu_instruction_proto);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VpuInstructionMsgObserver>, observer);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instruction_msg_link);
END_OBJECT_MSG(VpuInstructionMsg);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuInstruction);
  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuId, vpu_id);
  OBJECT_MSG_DEFINE_OPTIONAL(VpuInstructionMsg, vpu_instruction_msg);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instruction_link);

  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MirroredObjectAccess, vpu_instruction_oprand_index,
                                  oprand_index2object_access);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, vpu_instruction_oprand_ready_link,
                              ready_oprand_list);
END_OBJECT_MSG(VpuInstruction);
// clang-format on

class OBJECT_MSG_TYPE(Vpu);

// clang-format off
BEGIN_OBJECT_MSG(RunningVpuInstructionPackage);
 public:
  template<typename NewQuerierT>
  void __Init__(OBJECT_MSG_TYPE(Vpu)* vpu, const NewQuerierT& NewQuerier) {
    set_vpu(vpu);
    mutable_status_querier()->__Init__(NewQuerier);
  }
  bool Done() const { return status_querier()->Done(); }

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(Vpu)*, vpu);
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<VpuInstructionStatusQuerier>, status_querier); 
  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstruction, vpu_instruction_link, vpu_instruction_list);
  OBJECT_MSG_DEFINE_LIST_LINK(running_vpu_instruction_package_link);
END_OBJECT_MSG(RunningVpuInstructionPackage);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(Vpu);
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_FLAT_MSG_KEY(7, VpuId, vpu_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstruction, vpu_instruction_link, pending_vpu_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(RunningVpuInstructionPackage, running_vpu_instruction_package_link,
                              running_vpu_instruction_package_list);
END_OBJECT_MSG(Vpu);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuScheduler);
  //links
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionMsg, vpu_instruction_msg_link, vpu_instruction_msg_pending_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(Vpu, vpu_id, vpu_id2vpu);
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObject, logical_object_id, id2logical_object);
END_OBJECT_MSG(VpuScheduler);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_
