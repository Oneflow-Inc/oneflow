#ifndef ONEFLOW_CORE_KERNEL_SALR_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SALR_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class SALRMdUpdateKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SALRMdUpdateKernel);
  SALRMdUpdateKernel() = default;
  ~SALRMdUpdateKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }

 private:
};

template<DeviceType device_type, typename FloatingPointType>
class SALRMdUpdateKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SALRMdUpdateKernelUtil);
  SALRMdUpdateKernelUtil() = delete;

  static void UpdateLearningRate(const KernelCtx& ctx, const int64_t n,
                                 const FloatingPointType delta,
                                 FloatingPointType* last_diff_flag,
                                 const FloatingPointType* model_diff,
                                 FloatingPointType* learning_rate);

  static void UpdateModel(const KernelCtx& ctx, const int64_t n,
                          FloatingPointType* model,
                          const FloatingPointType* model_diff,
                          const FloatingPointType* learning_rate,
                          const FloatingPointType epsilon);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SALR_MODEL_UPDATE_KERNEL_H_
