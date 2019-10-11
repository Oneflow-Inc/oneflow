#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

#include "tensorflow/compiler/jit/xla_lib/xla_runtime_util.h"

namespace xla {

void SwapGpuStreamHandle(se::Stream *stream, void **gpu_stream) {
  void **cuda_stream = se::gpu::AsGpuStream(stream)->GpuStreamMemberHack();
  void *tmp_stream = *cuda_stream;
  *cuda_stream = *gpu_stream;
  *gpu_stream = tmp_stream;
}

inline size_t Align(int alignment, size_t size) {
  return (size + alignment - 1) / alignment * alignment;
}

gpu::GpuExecutable *AsGpuExecutable(Executable *executable) {
  return dynamic_cast<gpu::GpuExecutable *>(executable);
}

cpu::CpuExecutable *AsCpuExecutable(Executable *executable) {
  return dynamic_cast<cpu::CpuExecutable *>(executable);
}

const BufferAssignment *GetBufferAssignment(Executable *executable) {
  const BufferAssignment *assignment = nullptr;
  if (auto *gpu_executable = AsGpuExecutable(executable)) {
    assignment = (gpu_executable->GetBufferAssignment()).get();
  } else if (auto *cpu_executable = AsCpuExecutable(executable)) {
    assignment = &(cpu_executable->buffer_assignment());
  } else {
    LOG(FATAL) << "Only support CPU or GPU executable.";
  }
  return assignment;
}

size_t CalcWorkspaceByteSize(LocalExecutable *local_executable) {
  const BufferAssignment *assignment =
      GetBufferAssignment(local_executable->executable());
  CHECK_NOTNULL(assignment);

  size_t workspace_bytes = 0;
  for (int i = 0; i < assignment->Allocations().size(); ++i) {
    const BufferAllocation& allocation = assignment->GetAllocation(i);
    // if (!allocation.is_entry_computation_parameter()) {
    if (allocation.IsPreallocatedTempBuffer() || allocation.is_tuple()) {
      workspace_bytes += Align(64/*alignment*/, allocation.size());
    }
  }
  return workspace_bytes;
}

Status ResultAllocationIndices(LocalExecutable *local_executable,
                               std::vector<int64_t> *indices) {
  const BufferAssignment *assignment =
      GetBufferAssignment(local_executable->executable());
  CHECK_NOTNULL(assignment);

  std::unordered_map<int, int> allocation_order;
  for (int i = 0; i < assignment->Allocations().size(); ++i) {
    const BufferAllocation& allocation = assignment->GetAllocation(i);
    if ((allocation.maybe_live_out() ||
        allocation.IsPreallocatedTempBuffer()) &&
        !allocation.is_entry_computation_parameter()) {
      allocation_order.emplace(i, allocation_order.size());
    }
  }

  const HloModule &module = local_executable->executable()->module();
  const HloInstruction* root = module.entry_computation()->root_instruction();
  const InstructionValueSet &value_set =
      assignment->dataflow_analysis().GetInstructionValueSet(root);

  CHECK(root->shape().IsTuple());
  int tuple_size = root->shape().tuple_shapes_size();
  std::vector<int64_t> allocation_indices(tuple_size);
  for (int i = 0; i < tuple_size; ++i) {
    const auto& sources = value_set.element({i});
    CHECK_EQ(1, sources.values().size());
    auto instruction = sources.values()[0]->instruction();

    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        assignment->GetUniqueSlice(
                            instruction, sources.values()[0]->index()));
    if (slice.allocation()->is_entry_computation_parameter()) {
      // Output aliased with input will not be allocated, so we assign the
      // allocation index -1
      allocation_indices[i] = -1;
    } else {
      // Slice index is the allocation index
      allocation_indices[i] = allocation_order.at(slice.index());
    }
  }

  *indices = std::move(allocation_indices);
  return Status::OK();
}

}  // namespace xla
