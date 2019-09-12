#ifndef ONEFLOW_CORE_KERNEL_DATA_LOAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DATA_LOAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class DataLoadKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoadKernel);
  DataLoadKernel() = default;
  ~DataLoadKernel() = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void VirtualKernelInit() override;

  std::unique_ptr<data::DataLoader> data_loader_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DATA_LOAD_KERNEL_H_
