#ifndef ONEFLOW_CORE_KERNEL_SALR_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SALR_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class SALRMdUpdateKernel final : public ModelUpdtKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SALRMdUpdateKernel);
  SALRMdUpdateKernel() = default;
  ~SALRMdUpdateKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

  void InitDataTmpBlobs(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const override;

 private:
};

template<DeviceType device_type, typename FloatingPointType>
class SALRMdUpdateKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SALRMdUpdateKernelUtil);
  SALRMdUpdateKernelUtil() = delete;

  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          const FloatingPointType delta,
                          const FloatingPointType epsilon,
                          FloatingPointType* model,
                          const FloatingPointType* model_diff,
                          FloatingPointType* last_diff_flag,
                          FloatingPointType* learning_rate);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SALR_MODEL_UPDATE_KERNEL_H_
