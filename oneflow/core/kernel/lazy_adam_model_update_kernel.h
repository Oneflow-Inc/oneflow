#ifndef ONEFLOW_CORE_KERNEL_LAZY_ADAM_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LAZY_ADAM_MODEL_UDPATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LazyAdamMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyAdamMdUpdateKernel);
  LazyAdamMdUpdateKernel() = default;
  ~LazyAdamMdUpdateKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void UpdateModel(DeviceCtx* ctx, const T* batch_instance_num_ptr, T l1, T l2,
                   const int64_t* train_step, const float* learning_rate,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class LazyAdamMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const float* learning_rate, T l1, T l2, T beta1,
                          T beta2, T epsilon, const int64_t* train_step, T* beta1_t, T* beta2_t,
                          T* model_diff, T* model, T* m, T* v);
};

DECLARE_MDUPDT_KERNEL_CREATOR(LazyAdam);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LAZY_ADAM_MODEL_UPDATE_KERNEL_H_
