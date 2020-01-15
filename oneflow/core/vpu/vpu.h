#ifndef ONEFLOW_CORE_EAGER_VPU_H_
#define ONEFLOW_CORE_EAGER_VPU_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/vpu/vpu.pb.h"

namespace oneflow {

struct SymbolValue final {
  FLAT_MSG(SymbolValue);

  FLAT_MSG_DEFINE_FIELD(int64_t, value);
};

struct MakeSymbolInstruction final {
  FLAT_MSG(MakeSymbolInstruction);
};

struct RemoteCopyRange final {
  FLAT_MSG(RemoteCopyRange);

  FLAT_MSG_DEFINE_FIELD(int64_t, offset);
  FLAT_MSG_DEFINE_FIELD(int64_t, length);
};

struct NewHostBlobInstruction final {
  FLAT_MSG(NewHostBlobInstruction);

  FLAT_MSG_DEFINE_FIELD(SymbolValue, blob_desc_symbol);
};

struct DeleteHostBlobInstruction final {
  FLAT_MSG(DeleteHostBlobInstruction);

  FLAT_MSG_DEFINE_FIELD(SymbolValue, blob_desc_symbol);
};

struct HostMemoryVpu final {
  FLAT_MSG(HostMemoryVpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(MakeSymbolInstruction, make_symbol)
      FLAT_MSG_ONEOF_FIELD(NewHostBlobInstruction, new_host_blob)
      FLAT_MSG_ONEOF_FIELD(DeleteHostBlobInstruction, delete_host_blob));
  // clang-format on
};

struct DeviceMemoryVpu final {
  FLAT_MSG(DeviceMemoryVpu);
};

struct CpuDeviceVpu final {
  FLAT_MSG(CpuDeviceVpu);
};

struct GpuDeviceVpu final {
  FLAT_MSG(GpuDeviceVpu);
};

struct CopyHDInstruction final {
  FLAT_MSG(CopyHDInstruction);

  FLAT_MSG_DEFINE_FIELD(SymbolValue, src_blob_name_symbol);
  FLAT_MSG_DEFINE_FIELD(SymbolValue, dst_blob_name_symbol);
};

struct H2DTransportVpu final {
  FLAT_MSG(H2DTransportVpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(CopyHDInstruction, copy_h2d));
  // clang-format on
};

struct D2HTransportVpu final {
  FLAT_MSG(D2HTransportVpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(CopyHDInstruction, copy_d2h);
  // clang-format on
};

struct H2DTransportVpu final {
  FLAT_MSG(H2DTransportVpu);

  FLAT_MSG_DEFINE_FIELD(SymbolValue, src_blob_name_symbol);
  FLAT_MSG_DEFINE_FIELD(SymbolValue, dst_blob_name_symbol);
  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(range,
      FLAT_MSG_ONEOF_FIELD(RemoteCopyRange, contiguous_header_body_range)
      FLAT_MSG_ONEOF_FIELD(RemoteCopyRange, header_range)
      FLAT_MSG_ONEOF_FIELD(RemoteCopyRange, body_range));
  // clang-format on
};

// local host to remote host
struct L2RTransportVpu final {
  FLAT_MSG(L2RTransportVpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

// local host to remote host
struct L2RTransportLTE4096Vpu final {
  FLAT_MSG(L2RTransportLTE4096Vpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

// remote host to local host
struct R2LTransportVpu final {
  FLAT_MSG(R2LTransportVpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

// remote host to local host
struct R2LTransportLTE4096Vpu final {
  FLAT_MSG(R2LTransportLTE4096Vpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

struct HaltInstruction final {
  FLAT_MSG(HaltInstruction);
};

struct SyncInstruction final {
  FLAT_MSG(SyncInstruction);
};

struct ControlVpu final {
  FLAT_MSG(ControlVpu);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(instruction_type,
      FLAT_MSG_ONEOF_FIELD(SyncInstruction, sync)
      FLAT_MSG_ONEOF_FIELD(HaltInstruction, halt));
  // clang-format on
};

struct AllEnabledMask final {
  FLAT_MSG(AllEnabledMask);
};

struct VpuInstructionMask final {
  FLAT_MSG(VpuInstructionMask);

  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(mask_type,
      FLAT_MSG_ONEOF_FIELD(AllEnabledMask, all_enabled_mask)
      FLAT_MSG_ONEOF_FIELD(SymbolValue, parallel_desc_symbol));
  // clang-format on
};

struct VpuInstruction final {
  FLAT_MSG(VpuInstruction);

  FLAT_MSG_DEFINE_FIELD(VpuInstructionMask, instruction_mask);
  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(vpu_type,
      FLAT_MSG_ONEOF_FIELD(HostMemoryVpu, host_memory)
      FLAT_MSG_ONEOF_FIELD(DeviceMemoryVpu, device_memory)
      FLAT_MSG_ONEOF_FIELD(CpuDeviceVpu, cpu_device)
      FLAT_MSG_ONEOF_FIELD(GpuDeviceVpu, gpu_device)
      FLAT_MSG_ONEOF_FIELD(H2DTransportVpu, h2d_transport)
      FLAT_MSG_ONEOF_FIELD(D2HTransportVpu, d2h_transport)
      FLAT_MSG_ONEOF_FIELD(L2RTransportVpu, l2r_transport)
      FLAT_MSG_ONEOF_FIELD(R2LTransportVpu, r2l_transport)
      FLAT_MSG_ONEOF_FIELD(L2RTransportLTE4096Vpu, l2r_transport_lte4096)
      FLAT_MSG_ONEOF_FIELD(R2LTransportLTE4096Vpu, r2l_transport_lte4096)
      FLAT_MSG_ONEOF_FIELD(ControlVpu, control));
  // clang-format on
};

}  // namespace oneflow

#endif ONEFLOW_CORE_EAGER_VPU_H_
