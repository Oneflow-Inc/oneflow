#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_SCOPE_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_SCOPE_H_

#include "tensorflow/compiler/jit/xla_lib/swap_gpu_stream.h"
#include "tensorflow/compiler/jit/xla_lib/xla_runtime_util.h"
#include "oneflow/core/job/resource.pb.h"  // DeviceType
#include "oneflow/core/device/device_context.h"
#include "oneflow/engine/xla/of2xla/xla_utility.h"
#include "oneflow/engine/xla/of2xla/xla_launch_context.h"

namespace oneflow {
namespace mola {

class XlaLaunchScope {
 public:
  explicit XlaLaunchScope(xla::LocalExecutable *executable,
                          XlaLaunchContext *launch_ctx);

  virtual ~XlaLaunchScope();

 private:
  XlaLaunchContext *launch_ctx_;
  void **cuda_stream_;
};

XlaLaunchScope::XlaLaunchScope(xla::LocalExecutable *executable,
                               XlaLaunchContext *launch_ctx)
    : launch_ctx_(launch_ctx) {
  // Swap cuda stream between the backend stream and device context, so XLA
  // could launch kernel on the specified cuda stream of device context. Note
  // that it should do nothing for CPU mode in `SwapStreamHandle`
  if (launch_ctx_->device_type() == DeviceType::kGPU) {
    auto *device_ctx = launch_ctx_->device_ctx();
    cuda_stream_ = const_cast<void **>(
      reinterpret_cast<void * const*>(&(device_ctx->cuda_stream())));
    xla::SwapGpuStreamHandle(launch_ctx_->stream(), cuda_stream_);
  }

  size_t workspace_size = xla::CalcWorkspaceByteSize(executable);
  launch_ctx_->allocator()->ReserveWorkspace(workspace_size);

  launch_ctx_->allocator()->LockWorkspace();
}

XlaLaunchScope::~XlaLaunchScope() {
  // Swap again to let the subsequent cuda kernels use the original stream
  if (launch_ctx_->device_type() == DeviceType::kGPU) {
    xla::SwapGpuStreamHandle(launch_ctx_->stream(), cuda_stream_);
  }

  launch_ctx_->allocator()->UnlockWorkspace();
}

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_SCOPE_H_  
