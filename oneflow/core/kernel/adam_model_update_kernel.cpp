#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

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
    DeviceCtx* ctx, const T* batch_instance_num_ptr, T learning_rate, T l1, T l2,
    int64_t next_model_vid, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  const auto& adam_conf = GetAdamModelUpdateConf(this->op_conf());
  if ((next_model_vid != 1) && adam_conf.do_bias_correction()) {
    KernelUtil<device_type, T>::Scal(ctx, 1, static_cast<T>(adam_conf.beta1()),
                                     beta1_t_blob->mut_dptr<T>(), 1);
    KernelUtil<device_type, T>::Scal(ctx, 1, static_cast<T>(adam_conf.beta2()),
                                     beta2_t_blob->mut_dptr<T>(), 1);
  }
  KernelUtil<device_type, T>::Div(ctx, model_blob->shape().elem_cnt(),
                                  BnInOp2Blob("model_diff")->mut_dptr<T>(), batch_instance_num_ptr);
  AdamMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), learning_rate, l1, l2, static_cast<T>(adam_conf.beta1()),
      static_cast<T>(adam_conf.beta2()), static_cast<T>(adam_conf.epsilon()),
      adam_conf.do_bias_correction(), next_model_vid,
      (beta1_t_blob ? beta1_t_blob->dptr<T>() : nullptr),
      (beta2_t_blob ? beta2_t_blob->dptr<T>() : nullptr), BnInOp2Blob("model_diff")->mut_dptr<T>(),
      model_blob->mut_dptr<T>(), m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>());
}

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, T learning_rate, T l1, T l2, T beta1, T beta2,
                          T epsilon, bool do_bias_correction, int64_t next_model_vid,
                          const T* beta1_t, const T* beta2_t, T* model_diff, T* model, T* m, T* v) {
    // first-order moment
    UpdateMomentEstimate<T>(n, do_bias_correction, beta1, 1, model_diff, beta1_t, m);
    // second-order moment
    UpdateMomentEstimate<T>(n, do_bias_correction, beta2, 2, model_diff, beta2_t, v);
    FOR_RANGE(int64_t, i, 0, n) {
      model_diff[i] = m[i] / (std::sqrt(v[i]) + epsilon);
      T reg_diff = RegDiff(model_diff[i], l1, l2, model[i]);
      model[i] = model[i] - learning_rate * reg_diff;
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Adam);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdamModelUpdateConf, AdamMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
