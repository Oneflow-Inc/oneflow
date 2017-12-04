#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MomentumMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_acc_blob = BnInOp2Blob("model_diff_acc");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  float learning_rate = this->op_conf().momentum_mdupdt_conf().learning_rate();
  float beta = this->op_conf().momentum_mdupdt_conf().beta();
  if (*static_cast<const int64_t*>(ctx.other) == 1) { beta = 0.0f; }
  float alpha = learning_rate / JobDesc::Singleton()->BatchSize();
  CHECK(std::isfinite(alpha));

  // momentum = beta * momentum
  KernelUtil<device_type, T>::BlasScal(
      ctx.device_ctx, momentum_blob->shape().elem_cnt(), static_cast<T>(beta),
      momentum_blob->mut_dptr<T>(), 1);

  // momentum = momentum - alpha * model_diff_acc
  KernelUtil<device_type, T>::BlasAxpy(
      ctx.device_ctx, momentum_blob->shape().elem_cnt(), static_cast<T>(-alpha),
      model_diff_acc_blob->dptr<T>(), 1, momentum_blob->mut_dptr<T>(), 1);

  // model = model + momentum
  KernelUtil<device_type, T>::BlasAxpy(
      ctx.device_ctx, model_blob->shape().elem_cnt(), static_cast<T>(1),
      momentum_blob->dptr<T>(), 1, model_blob->mut_dptr<T>(), 1);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMomentumMdupdtConf,
                           MomentumMdUpdateKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
