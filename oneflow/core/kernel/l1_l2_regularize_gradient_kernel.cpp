#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/l1_l2_regularize_gradient_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class L1L2RegularizeGradientKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L1L2RegularizeGradientKernel);
  L1L2RegularizeGradientKernel() = default;
  ~L1L2RegularizeGradientKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
void L1L2RegularizeGradientKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const L1L2RegularizeGradientOpConf& conf = this->op_conf().l1_l2_regularize_gradient_conf();
  const Blob* model = BnInOp2Blob("model");
  const Blob* model_diff = BnInOp2Blob("model_diff");
  Blob* out = BnInOp2Blob("out");
  out->CopyDataContentFrom(ctx.device_ctx, model_diff);
  L1L2RegularizeGradientKernelUtil<device_type, T>::RegularizeGradient(
      ctx.device_ctx, out->shape().elem_cnt(), model->dptr<T>(), out->dptr<T>(), out->mut_dptr<T>(),
      conf.l1(), conf.l2());
}

template<DeviceType device_type, typename T>
const PbMessage& L1L2RegularizeGradientKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().l1_l2_regularize_gradient_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL1L2RegularizeGradientConf, L1L2RegularizeGradientKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
