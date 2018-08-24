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
  void UpdateModel(DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2,
                   const Blob* pre_model_blob, const Blob* model_diff_blob, int64_t next_model_vid,
                   std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class LARSMdUpdateKernelUtil final {
 public:
  static void SumOfSquare(DeviceCtx*, const int64_t n, const T* x, T* result);
};

DECLARE_MDUPDT_KERNEL_CREATOR(LARS);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LARS_MODEL_UPDATE_KERNEL_H_
