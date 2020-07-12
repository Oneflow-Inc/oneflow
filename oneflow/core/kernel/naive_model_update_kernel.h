#ifndef ONEFLOW_CORE_KERNEL_NAIVE_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NAIVE_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NaiveMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NaiveMdUpdateKernel);
  NaiveMdUpdateKernel() = default;
  ~NaiveMdUpdateKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void UpdateModel(DeviceCtx* ctx, T weight_decay, const int64_t* train_step,
                   const float* learning_rate,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  bool IsWeightDecaySupported() override { return true; }
};

template<DeviceType device_type, typename T>
class NaiveMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const float* learning_rate, T weight_decay,
                          const T* model_diff, T* model);
};

DECLARE_MDUPDT_KERNEL_CREATOR(Naive);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NAIVE_MODEL_UPDATE_KERNEL_H_
