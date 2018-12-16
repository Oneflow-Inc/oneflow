#ifndef ONEFLOW_CORE_KERNEL_CENTER_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CENTER_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class CenterLossKernel final : public LossKernel<device_type, PredType, LabelType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CenterLossKernel);
  CenterLossKernel() = default;
  ~CenterLossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithRandomSeed(
      DeviceCtx* ctx, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct CenterLossKernelUtil {
  static void Forward(DeviceCtx* ctx, const PredType* prediction, const LabelType* label,
                      const int32_t num_classes, const int32_t dim, const int32_t num_labels,
                      const float alpha, PredType* piece_centers, PredType* centers, PredType* loss,
                      PredType* prediction_diff);
};

};  // namespace oneflow

#endif
