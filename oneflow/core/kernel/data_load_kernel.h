#ifndef ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/dataset/data_loader.h"

namespace oneflow {

struct DataLoadStatus {
  bool is_eof;
};

class DataLoadKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoadKernel);
  DataLoadKernel() = default;
  ~DataLoadKernel() override = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  std::unique_ptr<DataLoader> data_loader_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RECORD_LOAD_KERNEL_H_
