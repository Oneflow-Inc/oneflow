#ifndef ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MdUpdateKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdateKernel);
  ~MdUpdateKernel() = default;

  void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 protected:
  MdUpdateKernel() = default;
  virtual void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, int64_t next_model_vid,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

 private:
  void L1Regularization(DeviceCtx* ctx, float weight_decay,
                        const Blob* model_blob,
                        Blob* model_diff_acc_blob) const;
  void L2Regularization(DeviceCtx* ctx, float weight_decay,
                        const Blob* model_blob,
                        Blob* model_diff_acc_blob) const;
  void Regularization(
      DeviceCtx* ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

template<DeviceType device_type, typename T>
class MdUpdateKernelUtil final {
 public:
  static void L1Regularization(DeviceCtx* ctx, const int64_t n,
                               float weight_decay, const T* model,
                               T* model_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_
