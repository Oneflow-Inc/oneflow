#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/regular_gradient_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RegularGradientKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegularGradientKernel);
  RegularGradientKernel() = default;
  ~RegularGradientKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
void RegularGradientKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const RegularGradientOpConf& conf = this->op_conf().regular_gradient_conf();
  const Blob* model = BnInOp2Blob("model");
  const Blob* model_diff = BnInOp2Blob("model_diff");
  Blob* out = BnInOp2Blob("out");
  out->CopyDataContentFrom(ctx.device_ctx, model_diff);
  RegularGradientKernelUtil<device_type, T>::RegularGradient(
      ctx.device_ctx, out->shape().elem_cnt(), model->dptr<T>(), out->dptr<T>(), out->mut_dptr<T>(),
      conf.l1_scale(), conf.l2_scale());
}

template<DeviceType device_type, typename T>
const PbMessage& RegularGradientKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().regular_gradient_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRegularGradientConf, RegularGradientKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
