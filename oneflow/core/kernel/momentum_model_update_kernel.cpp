#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MomentumMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2, const Blob* pre_model_blob,
    const Blob* model_diff_blob, int64_t next_model_vid,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  if (next_model_vid == 1) {
    Memset<device_type>(ctx, momentum_blob->mut_dptr<T>(), 0,
                        momentum_blob->ByteSizeOfDataContentField());
  }
  float momentum_beta = this->op_conf().normal_mdupdt_conf().user_conf().momentum_conf().beta();

  NormalMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), batch_size, static_cast<T>(momentum_beta), learning_rate,
      l1, l2, model_diff_blob->dptr<T>(), pre_model_blob->dptr<T>(), momentum_blob->mut_dptr<T>(),
      model_blob->mut_dptr<T>());
}

DEFINE_MDUPDT_KERNEL_CREATOR(Momentum);

}  // namespace oneflow
