#ifndef ONEFLOW_CORE_EAGER_VPU_H_
#define ONEFLOW_CORE_EAGER_VPU_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/pod_proto.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/vpu/vpu.pb.h"

namespace oneflow {

struct SymbolValue final {
  POD_PROTO_DEFINE_FIELD(int64_t, value);
};

struct MakeSymbolInstruction final {};

struct ReallocSymbolBufferInstruction final {
  POD_PROTO_DEFINE_FIELD(int64_t, size);
};

struct RemoteCopyRange final {
  POD_PROTO_DEFINE_FIELD(int64_t, offset);
  POD_PROTO_DEFINE_FIELD(int64_t, length);
};

struct NewHostBlobInstruction final {
  POD_PROTO_DEFINE_FIELD(SymbolValue, blob_desc_symbol);
};

struct DeleteHostBlobInstruction final {
  POD_PROTO_DEFINE_FIELD(SymbolValue, blob_desc_symbol);
};

struct HostMemoryVpu final {
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(MakeSymbolInstruction, make_symbol)
      POD_PROTO_ONEOF_FIELD(ReallocSymbolBufferInstruction, realloc_symbol_buffer)
      POD_PROTO_ONEOF_FIELD(NewHostBlobInstruction, new_host_blob)
      POD_PROTO_ONEOF_FIELD(DeleteHostBlobInstruction, delete_host_blob));
  // clang-format on
};

struct DeviceMemoryVpu final {};

struct CpuDeviceVpu final {
  VpuInstructionParallel parallel;
};

struct GpuDeviceVpu final {
  VpuInstructionParallel parallel;
};

struct CopyHDInstruction final {
  POD_PROTO_DEFINE_FIELD(SymbolValue, src_blob_name_symbol);
  POD_PROTO_DEFINE_FIELD(SymbolValue, dst_blob_name_symbol);
};

struct H2DTransportVpu final {
  VpuInstructionParallel parallel;
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(CopyHDInstruction, copy_h2d));
  // clang-format on
};

struct D2HTransportVpu final {
  VpuInstructionParallel parallel;
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(CopyHDInstruction, copy_d2h);
  // clang-format on
};

struct RemoteCopyInstruction final {
  POD_PROTO_DEFINE_FIELD(SymbolValue, src_blob_name_symbol);
  POD_PROTO_DEFINE_FIELD(SymbolValue, dst_blob_name_symbol);
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(range,
      POD_PROTO_ONEOF_FIELD(RemoteCopyRange, contiguous_header_body_range)
      POD_PROTO_ONEOF_FIELD(RemoteCopyRange, header_range)
      POD_PROTO_ONEOF_FIELD(RemoteCopyRange, body_range));
  // clang-format on
};

// local host to remote host
struct L2RTransportVpu final {
  VpuInstructionParallel parallel;
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

// local host to remote host
struct L2RTransportLTE4096Vpu final {
  VpuInstructionParallel parallel;
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

// remote host to local host
struct R2LTransportVpu final {
  VpuInstructionParallel parallel;
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

// remote host to local host
struct R2LTransportLTE4096Vpu final {
  VpuInstructionParallel parallel;
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(RemoteCopyInstruction, copy));
  // clang-format on
};

struct HaltInstruction final {};

struct SyncInstruction final {};

struct ControlVpu final {
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(SyncInstruction, sync)
      POD_PROTO_ONEOF_FIELD(HaltInstruction, halt));
  // clang-format on
};

struct AllEnabledMask final {};
struct VpuInstructionMask final {
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(mask_type,
      POD_PROTO_ONEOF_FIELD(AllEnabledMask, all_enabled_mask)
      POD_PROTO_ONEOF_FIELD(SymbolValue, parallel_desc_symbol));
  // clang-format on
};

struct VpuInstruction final {
  POD_PROTO_DEFINE_FIELD(VpuInstructionMask, instruction_mask);
  // clang-format off
  POD_PROTO_DEFINE_ONEOF(vpu_type,
      POD_PROTO_ONEOF_FIELD(HostMemoryVpu, host_memory)
      POD_PROTO_ONEOF_FIELD(DeviceMemoryVpu, device_memory)
      POD_PROTO_ONEOF_FIELD(CpuDeviceVpu, cpu_device)
      POD_PROTO_ONEOF_FIELD(GpuDeviceVpu, gpu_device)
      POD_PROTO_ONEOF_FIELD(H2DTransportVpu, h2d_transport)
      POD_PROTO_ONEOF_FIELD(D2HTransportVpu, d2h_transport)
      POD_PROTO_ONEOF_FIELD(L2RTransportVpu, l2r_transport)
      POD_PROTO_ONEOF_FIELD(R2LTransportVpu, r2l_transport)
      POD_PROTO_ONEOF_FIELD(L2RTransportLTE4096Vpu, l2r_transport_lte4096)
      POD_PROTO_ONEOF_FIELD(R2LTransportLTE4096Vpu, r2l_transport_lte4096)
      POD_PROTO_ONEOF_FIELD(ControlVpu, control));
  // clang-format on
};

}  // namespace oneflow

#endif ONEFLOW_CORE_EAGER_VPU_H_
