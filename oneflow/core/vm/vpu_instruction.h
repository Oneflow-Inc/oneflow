#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/mirrored_object.h"

namespace oneflow {

// clang-format off
BEGIN_FLAT_MSG(AllVpuEnabledMask);
END_FLAT_MSG(AllVpuEnabledMask);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(VpuMask);
  FLAT_MSG_DEFINE_ONEOF(mask_type,
      FLAT_MSG_ONEOF_FIELD(AllVpuEnabledMask, all_vpu_enabled)
      FLAT_MSG_ONEOF_FIELD(LogicalObjectPtrValue, enabled_parallel_desc_symbol));
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

static const int kVpuOprandLimit = 64;

// clang-format off
BEGIN_FLAT_MSG(VpuInstructionProto);
  FLAT_MSG_DEFINE_OPTIONAL(VpuInstructionDesc, vpu_instruction_desc);
  FLAT_MSG_DEFINE_REPEATED(LogicalObjectPtrValue, operand, kVpuOprandLimit);
END_FLAT_MSG(VpuInstructionProto);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(VpuInstruction);
  // fields
  OBJECT_MSG_DEFINE_FLAT_MSG(VpuInstructionDesc, vpu_instruction_desc);

  // links
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObjectAccess, vpu_operand_index, index2mirrored_obj_access);
END_OBJECT_MSG(VpuInstruction);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_
