#ifndef ONEFLOW_CORE_KERNEL_PROCESS_MODEL_DIFF_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROCESS_MODEL_DIFF_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ProcessModelDiffKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ProcessModelDiffKernel);
  ProcessModelDiffKernel() = default;
  ~ProcessModelDiffKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;

  decltype(make_tuple_from_sequence<7>()) tp_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROCESS_MODEL_DIFF_KERNEL_H_
