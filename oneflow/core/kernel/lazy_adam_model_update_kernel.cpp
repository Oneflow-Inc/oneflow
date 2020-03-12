#include "oneflow/core/kernel/lazy_adam_model_update_kernel.h"

namespace oneflow {

namespace {

const LazyAdamModelUpdateConf& GetLazyAdamModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.lazy_adam_model_update_conf().user_conf().lazy_adam_conf();
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& LazyAdamMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().lazy_adam_model_update_conf();
}

template<DeviceType device_type, typename T>
void LazyAdamMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  const auto& lazy_adam_conf = GetLazyAdamModelUpdateConf(this->op_conf());
  LazyAdamMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), learning_rate, weight_decay,
      static_cast<T>(lazy_adam_conf.beta1()), static_cast<T>(lazy_adam_conf.beta2()),
      static_cast<T>(lazy_adam_conf.epsilon()), train_step, beta1_t_blob->mut_dptr<T>(),
      beta2_t_blob->mut_dptr<T>(), BnInOp2Blob("model_diff")->mut_dptr<T>(),
      model_blob->mut_dptr<T>(), m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>());
}

template<typename T>
class LazyAdamMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T beta1, T beta2, T epsilon, const int64_t* train_step, T* beta1_t,
                          T* beta2_t, T* model_diff, T* model, T* m, T* v) {
    const float local_learning_rate = *learning_rate * std::sqrt(1 - (*beta2_t)) / (1 - (*beta1_t));
    for (int64_t i = 0; i < n; ++i) {
      T model_diff_val = model_diff[i];
      if (abs(model_diff_val) < 1e-12) { continue; }
      m[i] = beta1 * m[i] + (1 - beta1) * model_diff_val;
      v[i] = beta2 * v[i] + (1 - beta2) * model_diff_val * model_diff_val;
      model[i] = model[i] - local_learning_rate * m[i] / (std::sqrt(v[i]) + epsilon);
    }
    *beta1_t *= beta1;
    *beta2_t *= beta2;
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(LazyAdam);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLazyAdamModelUpdateConf, LazyAdamMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
