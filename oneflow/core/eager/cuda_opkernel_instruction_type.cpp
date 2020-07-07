#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
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

class CudaUserStatelessCallOpKernelInstructionType final
    : public UserStatelessCallOpKernelInstructionType {
 public:
  CudaUserStatelessCallOpKernelInstructionType() = default;
  ~CudaUserStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CudaUserStatelessCallOpKernelInstructionType>(
    "gpu.compute.UserStatelessCallOpKernel"));

class CudaSystemStatelessCallOpKernelInstructionType final
    : public SystemStatelessCallOpKernelInstructionType {
 public:
  CudaSystemStatelessCallOpKernelInstructionType() = default;
  ~CudaSystemStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CudaSystemStatelessCallOpKernelInstructionType>(
    "gpu.compute.SystemStatelessCallOpKernel"));

class CudaCopyH2DSystemStatelessCallOpKernelInstructionType final
    : public SystemStatelessCallOpKernelInstructionType {
 public:
  CudaCopyH2DSystemStatelessCallOpKernelInstructionType() = default;
  ~CudaCopyH2DSystemStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaCopyH2DStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CudaCopyH2DSystemStatelessCallOpKernelInstructionType>(
    "gpu.copy_h2d.SystemStatelessCallOpKernel"));

class CudaCopyD2HSystemStatelessCallOpKernelInstructionType final
    : public SystemStatelessCallOpKernelInstructionType {
 public:
  CudaCopyD2HSystemStatelessCallOpKernelInstructionType() = default;
  ~CudaCopyD2HSystemStatelessCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaCopyD2HStreamType;

  std::shared_ptr<MemoryCase> GetOutBlobMemCase(const DeviceType device_type,
                                                const int64_t device_id) const override {
    auto mem_case = std::make_shared<MemoryCase>();
    mem_case->mutable_host_mem()->mutable_cuda_pinned_mem()->set_device_id(device_id);
    return mem_case;
  }

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<CudaCopyD2HSystemStatelessCallOpKernelInstructionType>(
    "gpu.copy_d2h.SystemStatelessCallOpKernel"));

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

class GpuFeedBlobInstructionType final : public FeedBlobInstructionType {
 public:
  GpuFeedBlobInstructionType() = default;
  ~GpuFeedBlobInstructionType() override = default;

  using stream_type = vm::CudaStreamType;

 private:
  const char* device_tag() const override { return stream_type().device_tag(); }
};
COMMAND(vm::RegisterInstructionType<GpuFeedBlobInstructionType>("gpu.FeedBlob"));

}  // namespace eager
}  // namespace oneflow
