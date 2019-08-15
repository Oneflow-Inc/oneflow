#ifndef ONEFLOW_CORE_KERNEL_ADAM_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ADAM_MODEL_UDPATE_KERNEL_H_

#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AdamMdUpdateKernel final : public NormalMdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdamMdUpdateKernel);
  AdamMdUpdateKernel() = default;
  ~AdamMdUpdateKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void UpdateModel(DeviceCtx* ctx, const T* batch_instance_num_ptr, T learning_rate, T l1, T l2,
                   int64_t next_model_vid,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class AdamMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, T learning_rate, T l1, T l2, T beta1, T beta2,
                          T epsilon, bool do_bias_correction, int64_t next_model_vid,
                          const T* beta1_t, const T* beta2_t, T* model_diff, T* model, T* m, T* v);
};

DECLARE_MDUPDT_KERNEL_CREATOR(Adam);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADAM_MODEL_UPDATE_KERNEL_H_
