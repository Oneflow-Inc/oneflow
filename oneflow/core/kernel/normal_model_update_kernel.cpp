#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, const Blob* pre_model_blob, const Blob* model_diff_blob,
    int64_t next_model_vid,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  float learning_rate = this->op_conf().normal_mdupdt_conf().learning_rate();

  if (pre_model_blob != model_blob) {
    model_blob->CopyDataContentFrom<device_type>(ctx, pre_model_blob);
  }

  // model = model - alpha * model_diff
  KernelUtil<device_type, T>::Axpy(ctx, model_blob->shape().elem_cnt(),
                                   -learning_rate, model_diff_blob->dptr<T>(),
                                   1, model_blob->mut_dptr<T>(), 1);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalMdupdtConf,
                           NormalMdUpdateKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
