#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

template<DeviceType device_type, typename T>
void AdamMdUpdateKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& adam_conf = this->op_conf().normal_mdupdt_conf().user_conf().adam_conf();
  InitializerConf m_init_conf;
  InitializerConf v_init_conf;
  m_init_conf.mutable_constant_conf()->set_value(0.0f);
  v_init_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &m_init_conf, 0, BnInOp2Blob("m"));
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &v_init_conf, 0, BnInOp2Blob("v"));
  if (!adam_conf.do_bias_correction()) { return; }
  InitializerConf beta1_init_conf;
  InitializerConf beta2_init_conf;
  beta1_init_conf.mutable_constant_conf()->set_value(adam_conf.beta1());
  beta2_init_conf.mutable_constant_conf()->set_value(adam_conf.beta2());
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &beta1_init_conf, 0,
                                                       BnInOp2Blob("beta1_t"));
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &beta2_init_conf, 0,
                                                       BnInOp2Blob("beta2_t"));
}

template<DeviceType device_type, typename T>
void AdamMdUpdateKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& adam_conf = this->op_conf().normal_mdupdt_conf().user_conf().adam_conf();
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, m_blob, "m",
                                                m_blob->shape().At(0), m_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, v_blob, "v",
                                                v_blob->shape().At(0), v_blob->shape().Count(1));
  if (!adam_conf.do_bias_correction()) { return; }
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, beta1_t_blob, "beta1_t", beta1_t_blob->shape().At(0),
      beta1_t_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, beta2_t_blob, "beta2_t", beta2_t_blob->shape().At(0),
      beta2_t_blob->shape().Count(1));
}

template<DeviceType device_type, typename T>
void AdamMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2, int64_t next_model_vid,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  int64_t n = model_blob->shape().elem_cnt();
  const AdamModelUpdateConf& adam_conf =
      this->op_conf().normal_mdupdt_conf().user_conf().adam_conf();
  if ((next_model_vid != 1) && adam_conf.do_bias_correction()) {
    KernelUtil<device_type, T>::Scal(ctx, 1, static_cast<T>(adam_conf.beta1()),
                                     beta1_t_blob->mut_dptr<T>(), 1);
    KernelUtil<device_type, T>::Scal(ctx, 1, static_cast<T>(adam_conf.beta2()),
                                     beta2_t_blob->mut_dptr<T>(), 1);
  }
  KernelUtil<device_type, T>::Div(ctx, n, BnInOp2Blob("model_diff")->mut_dptr<T>(), batch_size);
  AdamMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, n, 1, learning_rate, l1, l2, static_cast<T>(adam_conf.beta1()),
      static_cast<T>(adam_conf.beta2()), static_cast<T>(adam_conf.epsilon()),
      adam_conf.do_bias_correction(), (beta1_t_blob ? beta1_t_blob->dptr<T>() : nullptr),
      (beta2_t_blob ? beta2_t_blob->dptr<T>() : nullptr), BnInOp2Blob("model_diff")->dptr<T>(),
      model_blob->mut_dptr<T>(), m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>());
}

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, int64_t batch_size, T learning_rate, T l1,
                          T l2, T beta1, T beta2, T epsilon, bool do_bias_correction,
                          const T* beta1_t, const T* beta2_t, const T* model_diff, T* model, T* m,
                          T* v) {
    FOR_RANGE(int64_t, i, 0, n) {
      m[i] = beta1 * m[i] + (1 - beta1) * model_diff[i];
      v[i] = beta2 * v[i] + (1 - beta2) * model_diff[i] * model_diff[i];
      if (do_bias_correction) {
        learning_rate = learning_rate * std::sqrt(1 - (*beta2_t)) / (1 - (*beta1_t));
      }
      T reg_diff = RegularizeDiff(m[i] / (std::sqrt(v[i]) + epsilon), batch_size, l1, l2, model[i]);
      model[i] = model[i] - learning_rate * reg_diff;
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Adam);

}  // namespace oneflow
