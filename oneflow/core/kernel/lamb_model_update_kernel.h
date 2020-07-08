#ifndef ONEFLOW_CORE_KERNEL_LAMB_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LAMB_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LAMBMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LAMBMdUpdateKernel);
  LAMBMdUpdateKernel() = default;
  ~LAMBMdUpdateKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void UpdateModel(DeviceCtx* ctx, T weight_decay, const int64_t* train_step,
                   const float* learning_rate,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  bool IsWeightDecaySupported() override { return true; }
};

template<DeviceType device_type, typename T>
class LAMBMdUpdateKernelUtil final {
 public:
  static void UpdateModel(
      DeviceCtx*, int64_t n, const float* learning_rate,
      T weight_decay, T beta1, T beta2, T epsilon, const int64_t* train_step,
      const T* beta1_t, const T* beta2_t, T* model_diff, T* model, T* m, T* v, T* fw_buf);
};

DECLARE_MDUPDT_KERNEL_CREATOR(LAMB);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LAMB_MODEL_UPDATE_KERNEL_H_
