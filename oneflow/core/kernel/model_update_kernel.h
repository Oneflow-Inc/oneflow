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
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    auto tpl = reinterpret_cast<std::tuple<int64_t, const Blob*>*>(ctx.other);
    Regularization(ctx.device_ctx, BnInOp2Blob);
    UpdateModel(ctx.device_ctx, std::get<1>(*tpl), std::get<0>(*tpl),
                [BnInOp2Blob](const std::string& bn) {
                  if (bn == "model_diff_acc"
                      && JobDesc::Singleton()->regularization_method() != kNone) {
                    return BnInOp2Blob("regularized_diff");
                  }
                  return BnInOp2Blob(bn);
                });
  }

 protected:
  MdUpdateKernel() = default;
  virtual void UpdateModel(
      DeviceCtx* ctx, const Blob* pre_model_blob, int64_t next_model_vid,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

 private:
  void L1Regularization(DeviceCtx* ctx, const Blob* model_blob,
                        const Blob* model_diff_acc_blob,
                        Blob* regularized_diff_blob) {
    regularized_diff_blob->CopyDataContentFrom(ctx, model_diff_acc_blob);
    KernelUtil<device_type, T>::Axpy(ctx, model_diff_acc_blob->elem_cnt(),
                                     weight_decay, model_blob->dptr<T>(), 1,
                                     regularized_diff_blob->mut_dptr<T>(), 1);
  }
  void L1Regularization(DeviceCtx* ctx, const Blob* model_blob,
                        const Blob* model_diff_acc_blob,
                        Blob* regularized_diff_blob) {
    KernelUtil<device_type, T>::Sign(ctx, model_diff_acc_blob->dptr<T>(),
                                     regularized_diff_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Axpy(ctx, model_diff_acc_blob->elem_cnt(),
                                     weight_decay, model_blob->dptr<T>(), 1,
                                     regularized_diff_blob->mut_dptr<T>(), 1);
  }
  void Regularization(DeviceCtx* ctx,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) {
    auto regularization_method = JobDesc::Singleton()->regularization_method();
    if (regularization_method == kNone) { return; }
    Blob* regularized_diff_blob = BnInOp2Blob("regularized_diff");
    const Blob* model_diff_acc_blob = BnInOp2Blob("model_diff_acc");
    const Blob* model_blob = BnInOp2Blob("model");
    if (regularization_method == kL1) {
      L1Regularization(ctx, model_diff_acc_blob, model_blob, regularized_diff_blob);
    } else if (regularization_method == kL2) {
      L2Regularization(ctx, model_diff_acc_blob, model_blob, regularized_diff_blob);
    } else {
      UNEXPECTED_RUN();
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_
