#ifndef ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernel final : public MdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropMdUpdateKernel);
  RMSPropMdUpdateKernel() = default;
  ~RMSPropMdUpdateKernel() = default;

 private:
  void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, int64_t next_model_vid,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernelUtil final {
 public:
  // alpha = (1 - decay_rate) / batch_size ^ 2
  // mean_square = alpha * model_diff_acc ^ 2 + decay_rate * mean_square
  // learning_rate = learning_rate / batch_size
  // model = pre_model - learning_rate * model_diff_acc / sqrt(mean_square +
  // epsilon)
  static void UpdateModel(DeviceCtx*, const int64_t n, const T alpha,
                          const T learning_rate, const T decay_rate,
                          const T epsilon, const T* pre_model, T* model,
                          T* mean_square, const T* model_diff_acc);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
