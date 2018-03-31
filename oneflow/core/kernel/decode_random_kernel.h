#ifndef ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_

#include "oneflow/core/kernel/decode_ofrecord_kernel.h"

namespace oneflow {


class DecodeRandomKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomKernel);
  DecodeRandomKernel() = default;
  ~DecodeRandomKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  int32_t GenNextRandomMaxColId();
  
  int32_t random_seed_;
  std::mt19937 max_col_id_gen_;
  std::uniform_int_distribution<int32_t> max_col_id_dis_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_
