#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

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
    DeviceCtx* ctx, const T* batch_instance_num_ptr, T learning_rate, T l1, T l2,
    int64_t next_model_vid, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  const AdamModelUpdateConf& adam_conf =
      this->op_conf().normal_mdupdt_conf().user_conf().adam_conf();
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

}  // namespace oneflow
