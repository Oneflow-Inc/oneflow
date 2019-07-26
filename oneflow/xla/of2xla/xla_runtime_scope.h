#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_RUNTIME_SCOPE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_RUNTIME_SCOPE_H_

#include "tensorflow/compiler/jit/xla_lib/swap_gpu_stream.h"
#include "oneflow/core/job/resource.pb.h"  // DeviceType
#include "oneflow/core/device/device_context.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/xla_launch_context.h"

namespace oneflow {
namespace mola {

class XlaRuntimeScope {
 public:
  explicit XlaRuntimeScope(XlaLaunchContext *launch_ctx)
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
  }

  virtual ~XlaRuntimeScope() {
    // Swap again to let the subsequent cuda kernels use the original stream
    if (launch_ctx_->device_type() == DeviceType::kGPU) {
      xla::SwapGpuStreamHandle(launch_ctx_->stream(), cuda_stream_);
    }
  }

 private:
  XlaLaunchContext *launch_ctx_;
  void **cuda_stream_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_RUNTIME_SCOPE_H_
