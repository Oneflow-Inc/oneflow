#include "oneflow/core/kernel/adam_model_update_kernel.h"

namespace oneflow {

namespace {

const AdamModelUpdateConf& GetAdamModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.adam_model_update_conf().user_conf().adam_conf();
};

template<typename T>
void UpdateMomentEstimate(int64_t n, bool do_bias_correction, T beta, int32_t p,
                          const T* model_diff, const T* beta_t, T* moment) {
  FOR_RANGE(int64_t, i, 0, n) {
    // Update biased moment estimate
    moment[i] = beta * moment[i] + (1 - beta) * std::pow(model_diff[i], p);
    if (do_bias_correction) {
      // Correct deviation of moment estimate
      moment[i] = moment[i] / (1 - *beta_t);
    }
  }
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& AdamMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().adam_model_update_conf();
}

template<DeviceType device_type, typename T>
void AdamMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  const auto& adam_conf = GetAdamModelUpdateConf(this->op_conf());
  if (adam_conf.do_bias_correction()) {
    AdamMdUpdateKernelUtil<device_type, T>::DoBiasCorrection(
        ctx, train_step, static_cast<T>(adam_conf.beta1()), static_cast<T>(adam_conf.beta2()),
        beta1_t_blob->mut_dptr<T>(), beta2_t_blob->mut_dptr<T>());
  }
  AdamMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), learning_rate, weight_decay,
      static_cast<T>(adam_conf.beta1()), static_cast<T>(adam_conf.beta2()),
      static_cast<T>(adam_conf.epsilon()), adam_conf.do_bias_correction(), train_step,
      (beta1_t_blob ? beta1_t_blob->dptr<T>() : nullptr),
      (beta2_t_blob ? beta2_t_blob->dptr<T>() : nullptr), BnInOp2Blob("model_diff")->dptr<T>(),
      model_blob->mut_dptr<T>(), m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>());
}

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T weight_decay,
                          T beta1, T beta2, T epsilon, bool do_bias_correction,
                          const int64_t* train_step, const T* beta1_t, const T* beta2_t,
                          const T* model_diff, T* model, T* m, T* v) {
    // first-order moment
    UpdateMomentEstimate<T>(n, do_bias_correction, beta1, 1, model_diff, beta1_t, m);
    // second-order moment
    UpdateMomentEstimate<T>(n, do_bias_correction, beta2, 2, model_diff, beta2_t, v);
    FOR_RANGE(int64_t, i, 0, n) {
      const T mdv = m[i] / (std::sqrt(v[i]) + epsilon);
      model[i] = model[i] - *learning_rate * (mdv + weight_decay * model[i]);
    }
  }
  static void DoBiasCorrection(DeviceCtx*, const int64_t* train_step, const T beta1, const T beta2,
                               T* beta1_t, T* beta2_t) {
    if (*train_step != 0) {
      *beta1_t *= beta1;
      *beta2_t *= beta2;
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Adam);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdamModelUpdateConf, AdamMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
