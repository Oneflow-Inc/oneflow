#ifndef ONEFLOW_CORE_KERNEL_LAYER_NORM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LAYER_NORM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LayerNormKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormKernel);
  LayerNormKernel() = default;
  ~LayerNormKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().layer_norm_conf();
  }
};

template<DeviceType device_type, typename T>
struct LayerNormKernelUtil {
  static void NormalizeForward(const DeviceCtx* ctx, const Blob* in, const Blob* scale,
                               const Blob* bias, double epsilon, Blob* out, Blob* mean,
                               Blob* inv_variance);
  static void NormalizeBackward(const DeviceCtx* ctx, const Blob* in, const Blob* scale,
                                const Blob* mean, const Blob* inv_variance, const Blob* out_diff,
                                double epsilon, Blob* in_diff, Blob* scale_diff, Blob* bias_diff);
};

template<DeviceType device_type, typename T>
struct LayerNormConstBufInitUtil {
  static void InitConstBufBlobsImpl(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                    uint32_t random_seed, Blob* blob);
};

template<DeviceType device_type>
struct LayerNormConstBufInitUtil<device_type, float16> {
  static void InitConstBufBlobsImpl(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                    uint32_t random_seed, Blob* blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LAYER_NORM_KERNEL_H_
