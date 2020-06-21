#ifndef ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class DecodeRandomKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomKernel);
  DecodeRandomKernel() = default;
  ~DecodeRandomKernel() = default;

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  void VirtualKernelInit() override;
  uint32_t GenNextRandomSeed() const;

  std::unique_ptr<std::mt19937> gen_;
  std::unique_ptr<std::uniform_int_distribution<uint32_t>> dis_;

  mutable bool is_init_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_RANDOM_KERNEL_H_
