#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/regularize_gradient_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RegularizeGradientKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegularizeGradientKernel);
  RegularizeGradientKernel() = default;
  ~RegularizeGradientKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
void RegularizeGradientKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const RegularizeGradientOpConf& conf = this->op_conf().regularize_gradient_conf();
  const Blob* model = BnInOp2Blob("model");
  const Blob* model_diff = BnInOp2Blob("model_diff");
  Blob* out = BnInOp2Blob("out");
  out->CopyDataContentFrom(ctx.device_ctx, model_diff);
  RegularizeGradientKernelUtil<device_type, T>::RegularizeGradient(
      ctx.device_ctx, out->shape().elem_cnt(), model->dptr<T>(), out->dptr<T>(), out->mut_dptr<T>(),
      conf.l1_scale(), conf.l2_scale());
}

template<DeviceType device_type, typename T>
const PbMessage& RegularizeGradientKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().regularize_gradient_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRegularizeGradientConf, RegularizeGradientKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
