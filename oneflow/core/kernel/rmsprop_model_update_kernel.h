#ifndef ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernel final : public ModelUpdtKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropMdUpdateKernel);
  RMSPropMdUpdateKernel() = default;
  ~RMSPropMdUpdateKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

  void InitDataTmpBlobs(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const override;

 private:
};

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernelUtil final {
 public:
  // alpha = (1 - decay_rate) / batch_size ^ 2
  // mean_square = alpha * model_diff ^ 2 + decay_rate * mean_square
  static void UpdateMeanSquare(const KernelCtx& ctx, const int64_t n,
                               const T alpha, const T decay_rate,
                               T* mean_square, const T* model_diff);

  // alpha = learning_rate / batch_size
  // model -= alpha * model_diff / sqrt(mean_square + epsilon)
  static void UpdateModel(const KernelCtx& ctx, const int64_t n, T* model,
                          const T* model_diff, const T* mean_square,
                          const T epsilon, const T alpha);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
