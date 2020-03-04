#include "oneflow/core/kernel/lars_model_update_kernel.h"

namespace oneflow {

namespace {

const LARSModelUpdateConf& GetLARSModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.lars_model_update_conf().user_conf().lars_conf();
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& LARSMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().lars_model_update_conf();
}

template<DeviceType device_type, typename T>
void LARSMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  Blob* data_tmp_blob = BnInOp2Blob("lars_data_tmp");
  const LARSModelUpdateConf& lars_conf = GetLARSModelUpdateConf(this->op_conf());
  Memset<device_type>(ctx, data_tmp_blob->mut_dptr<T>(), 0, data_tmp_blob->ByteSizeOfBlobBody());
  LARSMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), learning_rate, weight_decay,
      static_cast<T>(lars_conf.momentum_beta()), static_cast<T>(lars_conf.epsilon()),
      static_cast<T>(lars_conf.lars_coefficient()), train_step, model_diff_blob->dptr<T>(),
      model_blob->mut_dptr<T>(), momentum_blob->mut_dptr<T>(), data_tmp_blob->mut_dptr<T>());
}

template<typename T>
class LARSMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T momentum_beta, T epsilon, T lars_coefficient, const int64_t* train_step,
                          const T* model_diff, T* model, T* momentum, T* data_tmp) {
    T model_norm = 0;
    T model_diff_norm = 0;
    FOR_RANGE(int64_t, i, 0, n) {
      model_norm += model[i] * model[i];
      model_diff_norm += model_diff[i] * model_diff[i];
    }
    model_norm = std::sqrt(model_norm / n);
    model_diff_norm = std::sqrt(model_diff_norm / n);
    T local_learning_rate = 0;
    if (*train_step == 0) {
      local_learning_rate =
          *learning_rate * lars_coefficient * model_norm / (epsilon + model_diff_norm);
    } else {
      local_learning_rate = *learning_rate * lars_coefficient * model_norm
                            / (epsilon + model_diff_norm + weight_decay * model_norm);
    }
    FOR_RANGE(int64_t, i, 0, n) {
      T reg_diff = model_diff[i] + weight_decay * model[i];
      momentum[i] = momentum_beta * momentum[i] - local_learning_rate * reg_diff;
      model[i] = model[i] + momentum[i];
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(LARS);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLarsModelUpdateConf, LARSMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
