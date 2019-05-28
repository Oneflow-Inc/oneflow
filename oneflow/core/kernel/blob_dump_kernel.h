#ifndef ONEFLOW_CORE_KERNEL_BLOB_DUMP_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BLOB_DUMP_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class BlobDumpKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobDumpKernel);
  BlobDumpKernel() = default;
  ~BlobDumpKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

  mutable int64_t iter_;
  int64_t parallel_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BLOB_DUMP_KERNEL_H_
