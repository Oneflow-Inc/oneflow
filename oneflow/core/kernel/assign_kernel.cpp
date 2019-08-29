#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class AssignKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AssignKernel);
  AssignKernel() = default;
  ~AssignKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
void AssignKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("ref")->CopyValidDataContentFrom(ctx.device_ctx, BnInOp2Blob("value"));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAssignConf, DeviceType::kCPU,
                            AssignKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAssignConf, DeviceType::kGPU,
                            AssignKernel<DeviceType::kGPU>);

}  // namespace oneflow
