#ifndef ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernel final : public MdUpdateKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MomentumMdUpdateKernel);
  MomentumMdUpdateKernel() = default;
  ~MomentumMdUpdateKernel() = default;

 private:
  void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, int64_t next_model_vid,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class MomentumMdUpdateKernelUtil final {
 public:
  static void UpdateModel(DeviceCtx*, const int64_t n, const T beta,
                          const T alpha, const T* model_diff_acc,
                          const T* pre_model, T* momentum, T* model);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MOMENTUM_MODEL_UPDATE_KERNEL_H_
