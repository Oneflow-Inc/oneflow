#ifndef ONEFLOW_CORE_KERNEL_LARS_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LARS_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LARSMdUpdateKernel final : public ModelUpdtKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LARSMdUpdateKernel);
  LARSMdUpdateKernel() = default;
  ~LARSMdUpdateKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

  void InitDataTmpBlobs(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const override;

 private:
};

template<DeviceType device_type, typename T>
class LARSMdUpdateKernelUtil final {
 public:
  //  local_lr = lars_coefficient * ||model|| / ( ||model_diff|| + ||model||)
  //  momentum[i] = m * momentum[i] + local_lr
  //                + learning_rate * (model_diff[i] + weight_decay * model[i])
  //  model[i] = model[i] - momentum[i]
  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          const T lars_coefficient, const T learning_rate,
                          const T m, const T weight_decay, T* model,
                          T* momentum, T* temp, const T* model_diff);
};

}  // namespace oneflow

#endif  //  ONEFLOW_CORE_KERNEL_LARS_MODEL_UPDATE_KERNEL_H_
