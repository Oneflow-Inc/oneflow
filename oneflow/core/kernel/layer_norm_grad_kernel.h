#ifndef ONEFLOW_CORE_KERNEL_LAYER_NORM_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LAYER_NORM_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LayerNormGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormGradKernel);
  LayerNormGradKernel() = default;
  ~LayerNormGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().layer_norm_grad_conf();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LAYER_NORM_GRAD_KERNEL_H_
