#ifndef ONEFLOW_CORE_EAGER_VPU_H_
#define ONEFLOW_CORE_EAGER_VPU_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/pod_proto.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/vpu/vpu.pb.h"

namespace oneflow {

// the smaller, the higher priority VpuSet is
enum VpuSet {
  kSymbolVpuSet = 0,
  kHostMemoryVpuSet,
  kDeviceVpuSet,
  kHaltVpuSet,
};

template<VpuSet vpu_set>
struct VpuPriority final {
  static const int value = vpu_set;
};

template<VpuSet vpu_set>
struct VpuParallelTrait final { };
template<>
struct VpuParallelTrait<VpuSet::kSymbolVpuSet> final {
  static const bool partial_enabled = false;
};

template<>
struct VpuParallelTrait<VpuSet::kHostMemoryVpuSet> final {
  static const bool partial_enabled = true;
};

template<>
struct VpuParallelTrait<VpuSet::kDeviceVpuSet> final {
  static const bool partial_enabled = true;
};

template<>
struct VpuParallelTrait<VpuSet::kHaltVpuSet> final {
  static const bool partial_enabled = false;
};

struct SymbolValue final {
  POD_PROTO_DEFINE_FIELD(int64_t, value);
};

struct MakeSymbolInstruction final {
};

struct ReallocSymbolBufferInstruction final {
  POD_PROTO_DEFINE_FIELD(int64_t, size);
};

struct SymbolFactoryVpu final {
  static const vpu_set = kSymbolVpuSet;
// clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(MakeSymbolInstruction, make_symbol)
      POD_PROTO_ONEOF_FIELD(ReallocSymbolBufferInstruction, realloc_symbol_buffer));
// clang-format on
};

struct RemoteCopyRange final {
  POD_PROTO_DEFINE_FIELD(int64_t, offset);
  POD_PROTO_DEFINE_FIELD(int64_t, length);
};

struct SymbolCopierVpu final {
  static const vpu_set = kSymbolVpuSet;
// clang-format off
  POD_PROTO_DEFINE_ONEOF(range,
      POD_PROTO_ONEOF_FIELD(RemoteCopyRange, copy_range));
// clang-format on
};

struct SmallSymbolCopierVpu final {
  static const vpu_set = kSymbolVpuSet;
};

struct NewHostBlobInstruction final {
  POD_PROTO_DEFINE_FIELD(SymbolValue, blob_desc_symbol);
};

struct DeleteHostBlobInstruction final {
  POD_PROTO_DEFINE_FIELD(SymbolValue, blob_desc_symbol);
};

struct HostMemoryVpu final {
  static const vpu_set = kHostMemoryVpuSet;
// clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(NewHostBlobInstruction, new_host_blob)
      POD_PROTO_ONEOF_FIELD(DeleteHostBlobInstruction, delete_host_blob));
// clang-format on
};

struct MemoryBarrierVpu final {
  static const vpu_set = kHostMemoryVpuSet;
};

struct CpuDeviceVpu final {
  static const vpu_set = kDeviceVpuSet;
  VpuInstructionParallel parallel;
};

struct GpuDeviceVpu final {
  static const vpu_set = kDeviceVpuSet;
  VpuInstructionParallel parallel;
};

struct CopyHDInstruction final {
  POD_PROTO_DEFINE_FIELD(SymbolValue, src_blob_name_symbol);
  POD_PROTO_DEFINE_FIELD(SymbolValue, dst_blob_name_symbol);
};

struct H2DCopierVpu final {
  static const vpu_set = kDeviceVpuSet;
  VpuInstructionParallel parallel;
// clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(CopyHDInstruction, copy_h2d));
// clang-format on
};

struct D2HCopierVpu final {
  static const vpu_set = kDeviceVpuSet;
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
struct L2RPusherVpu final {
  static const vpu_set = kDeviceVpuSet;
  VpuInstructionParallel parallel;
// clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(RemoteCopyInstruction, copy));
// clang-format on
};

// remote host to local host
struct R2LPullerVpu final {
  static const vpu_set = kDeviceVpuSet;
  VpuInstructionParallel parallel;
// clang-format off
  POD_PROTO_DEFINE_ONEOF(instruction_type,
      POD_PROTO_ONEOF_FIELD(RemoteCopyInstruction, copy));
// clang-format on
};

struct HaltVpu final {
  static const vpu_set = kHaltVpuSet;
};

struct AllVpuParallel final {};

struct VpuInstructionParallel final {
// clang-format off
  POD_PROTO_DEFINE_ONEOF(parallel_type,
      POD_PROTO_ONEOF_FIELD(AllVpuParallel, all_vpu)
      POD_PROTO_ONEOF_FIELD(SymbolValue, parallel_desc_symbol));
// clang-format on
};

struct VpuInstruction final {
// clang-format off
  POD_PROTO_DEFINE_ONEOF(vpu_type,
      POD_PROTO_ONEOF_FIELD(SymbolFactoryVpu, symbol_factory)
      POD_PROTO_ONEOF_FIELD(SymbolCopierVpu, symbol_copier)
      POD_PROTO_ONEOF_FIELD(SmallSymbolCopierVpu, small_symbol_copier)
      POD_PROTO_ONEOF_FIELD(HostMemoryVpu, host)
      POD_PROTO_ONEOF_FIELD(MemoryBarrierVpu, memory_barrier)
      POD_PROTO_ONEOF_FIELD(CpuDeviceVpu, cpu_device)
      POD_PROTO_ONEOF_FIELD(GpuDeviceVpu, gpu_device)
      POD_PROTO_ONEOF_FIELD(H2DCopierVpu, h2d_copier)
      POD_PROTO_ONEOF_FIELD(D2HCopierVpu, d2h_copier)
      POD_PROTO_ONEOF_FIELD(L2RPusherVpu, l2r_pusher)
      POD_PROTO_ONEOF_FIELD(R2LPullerVpu, r2l_puller)
      POD_PROTO_ONEOF_FIELD(HaltVpu, halt));
// clang-format on
};

}

#endif ONEFLOW_CORE_EAGER_VPU_H_
