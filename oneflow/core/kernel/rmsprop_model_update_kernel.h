#ifndef ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
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

template<DeviceType device_type, typename FloatingPointType>
class RMSPropMdUpdateKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropMdUpdateKernelUtil);
  RMSPropMdUpdateKernelUtil() = delete;

  // alpha = (1 - decay_rate) / batch_size ^ 2
  // mean_square = alpha * model_diff ^ 2 + decay_rate * mean_square
  static void UpdateMeanSquare(const KernelCtx& ctx, const int64_t n,
                               const FloatingPointType alpha,
                               const FloatingPointType decay_rate,
                               FloatingPointType* mean_square,
                               const FloatingPointType* model_diff);

  // alpha = learning_rate / batch_size
  // model -= alpha * model_diff / sqrt(mean_square + epsilon)
  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          FloatingPointType* model,
                          const FloatingPointType* model_diff,
                          const FloatingPointType* mean_square,
                          const FloatingPointType epsilon,
                          const FloatingPointType alpha);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
