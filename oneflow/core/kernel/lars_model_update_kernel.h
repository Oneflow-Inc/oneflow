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
  void UpdateModel(DeviceCtx* ctx, const int32_t* total_instance_num_ptr, T learning_rate, T l1,
                   T l2, int64_t next_model_vid,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class LARSMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const int32_t* total_instance_num_ptr,
                          T learning_rate, T l1, T l2, T momentum_beta, T epsilon,
                          T lars_coefficient, int64_t next_model_vid, const T* model_diff, T* model,
                          T* momentum, T* data_tmp);
};

DECLARE_MDUPDT_KERNEL_CREATOR(LARS);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LARS_MODEL_UPDATE_KERNEL_H_
