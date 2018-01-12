#ifndef ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormalMdUpdateKernel final : public MdUpdateKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdateKernel);
  NormalMdUpdateKernel() = default;
  ~NormalMdUpdateKernel() = default;

 private:
  void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, const Blob* model_diff_blob,
      int64_t next_model_vid,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
