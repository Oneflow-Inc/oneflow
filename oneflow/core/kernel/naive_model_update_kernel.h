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
  void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, const Blob* model_diff_blob,
      int64_t next_model_vid, double learning_rate,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

DECLARE_MDUPDT_KERNEL_CREATOR(Naive);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NAIVE_MODEL_UPDATE_KERNEL_H_
