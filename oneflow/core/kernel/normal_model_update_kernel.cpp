#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, const Blob* pre_model_blob, int64_t next_model_vid,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_acc_blob = BnInOp2Blob("model_diff_acc");
  Blob* model_blob = BnInOp2Blob("model");
  float learning_rate = this->op_conf().normal_mdupdt_conf().learning_rate();
  float alpha = learning_rate / JobDesc::Singleton()->BatchSize();
  CHECK(std::isfinite(alpha));

  if (pre_model_blob != model_blob) {
    model_blob->CopyDataContentFrom<device_type>(ctx, pre_model_blob);
  }

  // model = model - alpha * model_diff
  KernelUtil<device_type, T>::Axpy(ctx, model_blob->shape().elem_cnt(), -alpha,
                                   model_diff_acc_blob->dptr<T>(), 1,
                                   model_blob->mut_dptr<T>(), 1);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalMdupdtConf,
                           NormalMdUpdateKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
