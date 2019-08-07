#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

#include "tensorflow/compiler/jit/xla_lib/runtime_workspace_bytes.h"

namespace xla {

inline size_t Align(int alignment, size_t size) {
  return (size + alignment - 1) / alignment * alignment;
}

gpu::GpuExecutable *AsGpuExecutable(Executable *executable) {
  return dynamic_cast<gpu::GpuExecutable *>(executable);
}

cpu::CpuExecutable *AsCpuExecutable(Executable *executable) {
  return dynamic_cast<cpu::CpuExecutable *>(executable);
}

size_t CalcWorkspaceByteSize(LocalExecutable *local_executable) {
  Executable *executable = local_executable->executable();
  const BufferAssignment *assignment = nullptr;
  if (auto *gpu_executable = AsGpuExecutable(executable)) {
    assignment = (gpu_executable->GetBufferAssignment()).get();
  } else if (auto *cpu_executable = AsCpuExecutable(executable)) {
    assignment = &(cpu_executable->buffer_assignment());
  } else {
    LOG(FATAL) << "Only support CPU or GPU executable.";
  }
  CHECK_NOTNULL(assignment);

  size_t workspace_bytes = 0;
  for (int i = 0; i < assignment->Allocations().size(); ++i) {
    const BufferAllocation& allocation = assignment->GetAllocation(i);
    if (!allocation.is_entry_computation_parameter()) {
      workspace_bytes += Align(64/*alignment*/, allocation.size());
    }
  }
  return workspace_bytes;
}

}  // namespace xla
