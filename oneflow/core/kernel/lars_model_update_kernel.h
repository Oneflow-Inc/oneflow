#ifndef ONEFLOW_CORE_KERNEL_LARS_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LARS_MODEL_UDPATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LARSMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LARSMdUpdateKernel);
  LARSMdUpdateKernel() = default;
  ~LARSMdUpdateKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void UpdateModel(DeviceCtx* ctx, const T* batch_instance_num_ptr, T l1, T l2,
                   const int64_t* train_step, const float* learning_rate,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class LARSMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const T* batch_instance_num_ptr,
                          const float* learning_rate, T l1, T l2, T momentum_beta, T epsilon,
                          T lars_coefficient, const int64_t* train_step, const T* model_diff,
                          T* model, T* momentum, T* data_tmp);
};

DECLARE_MDUPDT_KERNEL_CREATOR(LARS);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LARS_MODEL_UPDATE_KERNEL_H_
