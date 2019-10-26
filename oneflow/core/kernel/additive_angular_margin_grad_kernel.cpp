#include "oneflow/core/kernel/additive_angular_margin_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AdditiveAngularMarginGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginGradKernel);
  AdditiveAngularMarginGradKernel() = default;
  ~AdditiveAngularMarginGradKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
const PbMessage& AdditiveAngularMarginGradKernel<device_type, T>::GetCustomizedOpConf() const {
  if (this->op_conf().has_additive_angular_margin_grad_conf()) {
    return this->op_conf().additive_angular_margin_grad_conf();
  } else if (this->op_conf().has_additive_angular_margin_ms1_grad_conf()) {
    return this->op_conf().additive_angular_margin_ms1_grad_conf();
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void AdditiveAngularMarginGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const float margin = GetValFromPbMessage<float>(this->GetCustomizedOpConf(), "margin");
  int64_t lower_bound = 0;
  if (this->kernel_conf().has_additive_angular_margin_grad_conf()) {
    lower_bound = this->kernel_conf().additive_angular_margin_grad_conf().lower_bound();
  }
  BnInOp2Blob("dx")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("dy"));
  AdditiveAngularMarginKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, lower_bound, static_cast<T>(cos(margin)), static_cast<T>(sin(margin)),
      BnInOp2Blob("dy"), BnInOp2Blob("label"), BnInOp2Blob("sin_theta_data"), BnInOp2Blob("dx"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdditiveAngularMarginGradConf,
                           AdditiveAngularMarginGradKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdditiveAngularMarginMs1GradConf,
                           AdditiveAngularMarginGradKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
