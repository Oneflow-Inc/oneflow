#ifndef ONEFLOW_CORE_KERNEL_DECODE_IN_STREAM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_IN_STREAM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class DecodeInStreamKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeInStreamKernel);
  DecodeInStreamKernel() = default;
  ~DecodeInStreamKernel() = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  //  void VirtualKernelInit(const ParallelContext*) override;
  mutable bool is_init_;
  std::ifstream file_;
  int64_t cur_read_size_;
  int64_t total_size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_IN_STREAM_KERNEL_H_