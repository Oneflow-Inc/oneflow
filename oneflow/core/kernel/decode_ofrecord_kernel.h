#ifndef ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

struct DecodeStatus {
  int32_t cur_col_id_;
  int32_t max_col_id_;
};

class DecodeOFRecordKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOFRecordKernel);
  DecodeOFRecordKernel() = default;
  ~DecodeOFRecordKernel() = default;

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  void VirtualKernelInit() override;
  int32_t NextRandomInt() const;

  std::unique_ptr<std::mt19937> random_seed_gen_;
  std::unique_ptr<std::uniform_int_distribution<int32_t>> distribution_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
