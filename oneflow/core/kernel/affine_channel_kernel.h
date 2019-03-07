#ifndef ONEFLOW_CORE_KERNEL_AFFINE_CHANNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_AFFINE_CHANNEL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AffineChannelKernel final : public KernelIfWithModel<device_type, T>,
                                  public KernelIfWithActivation<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AffineChannelKernel);
  AffineChannelKernel() = default;
  ~AffineChannelKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  bool HasSameShapeBetweenInOut() const { return true; }
  void InitModelBlobsWithRandomSeed(
      DeviceCtx* ctx, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class AffineChannelKernelUtil final {
 public:
  static void Forward(DeviceCtx* ctx, const int32_t elem_cnt, const int32_t channel_dim,
                      const int32_t channel_stride, const T* in, const T* scale, const T* bias,
                      T* out);
  static void BackwardInDiff(DeviceCtx* ctx, const int32_t elem_cnt, const int32_t channel_dim,
                             const int32_t channel_stride, const T* out_diff, const T* scale,
                             T* in_diff);
  static void BackwardScaleBiasDiff(DeviceCtx* ctx, const int32_t elem_cnt,
                                    const int32_t channel_dim, const int32_t channel_stride,
                                    const T* in, const T* out_diff, T* scale_diff, T* bias_diff);
  static void BackwardScaleDiff(DeviceCtx* ctx, const int32_t elem_cnt, const int32_t channel_dim,
                                const int32_t channel_stride, const T* in, const T* out_diff,
                                T* scale_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_AFFINE_CHANNEL_KERNEL_H_
