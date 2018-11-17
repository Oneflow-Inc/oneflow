#ifndef ONEFLOW_CORE_KERNEL_DEBUG_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DEBUG_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<typename T>
class DebugKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DebugKernel);
  DebugKernel() = default;
  ~DebugKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void InitOutStream(std::unique_ptr<PersistentOutStream>* out_stream,
                     const ParallelContext* parallel_ctx, const std::string& dir);
  void VirtualKernelInit(const ParallelContext* parallel_ctx);

  std::unique_ptr<PersistentOutStream> in_blob_out_stream_;
  std::unique_ptr<PersistentOutStream> out_diff_blob_out_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DEBUG_KERNEL_H_
