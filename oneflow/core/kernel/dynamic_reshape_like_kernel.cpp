#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class DynamicReshapeLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicReshapeLikeKernel);
  DynamicReshapeLikeKernel() = default;
  ~DynamicReshapeLikeKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
void DynamicReshapeLikeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("x");
  Blob* out_blob = BnInOp2Blob("y");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDynamicReshapeLikeConf, DynamicReshapeLikeKernel);

}  // namespace oneflow
