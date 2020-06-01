#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

class CudaCallOpKernelInstructionType final : public CallOpKernelInstructionType {
 public:
  CudaCallOpKernelInstructionType() = default;
  ~CudaCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CudaCallOpKernelInstructionType>("gpu.CallOpKernel"));

class CudaStatelessCallOpKernelInstructionType final : public StatelessCallOpKernelInstructionType {
 public:
  CudaStatelessCallOpKernelInstructionType() = default;
  ~CudaStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CudaStatelessCallOpKernelInstructionType>(
    "gpu.compute.StatelessCallOpKernel"));

class CudaDeprecatedStatelessCallOpKernelInstructionType final
    : public DeprecatedStatelessCallOpKernelInstructionType {
 public:
  CudaDeprecatedStatelessCallOpKernelInstructionType() = default;
  ~CudaDeprecatedStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CudaDeprecatedStatelessCallOpKernelInstructionType>(
    "gpu.compute.DeprecatedStatelessCallOpKernel"));

class GpuWatchBlobHeaderInstructionType final : public WatchBlobHeaderInstructionType {
 public:
  GpuWatchBlobHeaderInstructionType() = default;
  ~GpuWatchBlobHeaderInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<GpuWatchBlobHeaderInstructionType>("gpu.WatchBlobHeader"));

class GpuWatchBlobBodyInstructionType final : public WatchBlobBodyInstructionType {
 public:
  GpuWatchBlobBodyInstructionType() = default;
  ~GpuWatchBlobBodyInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<GpuWatchBlobBodyInstructionType>("gpu.WatchBlobBody"));

}  // namespace eager
}  // namespace oneflow
