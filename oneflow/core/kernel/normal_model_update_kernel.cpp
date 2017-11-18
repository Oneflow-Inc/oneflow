#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  const Blob* model_diffs_blob = BnInOp2Blob("model_diffs");
  float learning_rate = op_conf().normal_mdupdt_conf().learning_rate();
  float alpha = learning_rate / JobDesc::Singleton()->BatchSize();
  CHECK(std::isfinite(alpha));

  // model = model - alpha * model_diff
  KernelUtil<device_type, T>::BlasAxpy(
      ctx.device_ctx, model_blob->shape().elem_cnt(), -alpha,
      model_diffs_blob->dptr<T>(), 1, model_blob->mut_dptr<T>(), 1);
}

}  // namespace oneflow
