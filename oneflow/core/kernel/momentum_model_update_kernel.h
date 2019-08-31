#ifndef ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MomentumMdUpdateKernel);
  MomentumMdUpdateKernel() = default;
  ~MomentumMdUpdateKernel() = default;

 protected:
  const PbMessage& GetCustomizedOpConf() const override;

 private:
  void UpdateModel(DeviceCtx* ctx, const T* batch_instance_num_ptr, T l1, T l2,
                   const int64_t* global_step, const float* learning_rate,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const T* batch_instance_num_ptr, T beta,
                          const int64_t* global_step, const float* learning_rate, T l1, T l2,
                          const T* model_diff, T* model, T* momentum);
};

DECLARE_MDUPDT_KERNEL_CREATOR(Momentum);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
