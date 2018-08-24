#include "oneflow/core/kernel/lars_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LARSMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2, const Blob* pre_model_blob,
    const Blob* model_diff_blob, int64_t next_model_vid,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  if (next_model_vid == 1) {
    Memset<device_type>(ctx, momentum_blob->mut_dptr<T>(), 0,
                        momentum_blob->ByteSizeOfDataContentField());
  }
  const LARSModelUpdateConf& lars_conf =
      this->op_conf().normal_mdupdt_conf().user_conf().lars_conf();
  int64_t n = model_blob->shape().elem_cnt();
  T model_norm = 0;
  T model_diff_norm = 0;
  LARSMdUpdateKernelUtil<device_type, T>::SumOfSquare(ctx, n, pre_model_blob->dptr<T>(),
                                                      &model_norm);
  LARSMdUpdateKernelUtil<device_type, T>::SumOfSquare(ctx, n, model_diff_blob->dptr<T>(),
                                                      &model_diff_norm);
  model_norm = std::sqrt(model_norm / n);
  model_diff_norm = std::sqrt(model_diff_norm / n);
  T local_learning_rate = 0;
  if (next_model_vid == 1) {
    local_learning_rate = learning_rate * lars_conf.lars_coefficient() * model_norm
                          / (lars_conf.epsilon() + model_diff_norm);
  } else {
    local_learning_rate = learning_rate * lars_conf.lars_coefficient() * model_norm
                          / (lars_conf.epsilon() + model_diff_norm + l2 * model_norm);
  }
  NormalMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, n, batch_size, static_cast<T>(lars_conf.momentum_beta()), local_learning_rate, l1, l2,
      model_diff_blob->dptr<T>(), pre_model_blob->dptr<T>(), momentum_blob->mut_dptr<T>(),
      model_blob->mut_dptr<T>());
}

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void SumOfSquare(DeviceCtx*, const int64_t n, const T* x, T* result) {
    FOR_RANGE(int64_t, i, 0, n) { *result += x[i] * x[i]; }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(LARS);

}  // namespace oneflow
