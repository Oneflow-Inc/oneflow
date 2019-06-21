#ifndef ONEFLOW_CORE_KERNEL_LAYER_NORM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LAYER_NORM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LayerNormKernel final : public KernelIfWithModel<device_type, T>,
                              public KernelIfWithActivation<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormKernel);
  LayerNormKernel() = default;
  ~LayerNormKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithRandomSeed(DeviceCtx*, std::mt19937*,
                                    std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LAYER_NORM_KERNEL_H_
