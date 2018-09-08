#ifndef ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropMdUpdateKernel);
  RMSPropMdUpdateKernel() = default;
  ~RMSPropMdUpdateKernel() = default;

 private:
  void UpdateModel(DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2,
                   const Blob* pre_model_blob, int64_t next_model_vid,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class RMSPropMdUpdateKernelUtil final {
 public:
  // mean_square = (1 - decay_rate) * model_diff ^ 2 + decay_rate * mean_square
  // model = pre_model - learning_rate * model_diff / sqrt(mean_square + epsilon)
  static void UpdateModel(DeviceCtx*, int64_t n, int64_t batch_size, T learning_rate, T decay_rate,
                          T epsilon, T l1, T l2, const T* pre_model, T* model, T* mean_square,
                          const T* model_diff);
};

DECLARE_MDUPDT_KERNEL_CREATOR(RMSProp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RMSPROP_MODEL_UPDATE_KERNEL_H_
