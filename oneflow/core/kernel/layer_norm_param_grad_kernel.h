#ifndef ONEFLOW_CORE_KERNEL_LAYER_NORM_PARAM_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LAYER_NORM_PARAM_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LayerNormParamGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormParamGradKernel);
  LayerNormParamGradKernel() = default;
  ~LayerNormParamGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().layer_norm_param_grad_conf();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LAYER_NORM_PARAM_GRAD_KERNEL_H_
