#include "oneflow/core/kernel/lamb_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LAMBMdUpdateKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& lamb_conf = this->op_conf().normal_mdupdt_conf().user_conf().lamb_conf();
  InitializerConf m_init_conf;
  InitializerConf v_init_conf;
  m_init_conf.mutable_constant_conf()->set_value(0.0f);
  v_init_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &m_init_conf, 0, BnInOp2Blob("m"));
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &v_init_conf, 0, BnInOp2Blob("v"));
  InitializerConf beta1_init_conf;
  InitializerConf beta2_init_conf;
  beta1_init_conf.mutable_constant_conf()->set_value(lamb_conf.beta1());
  beta2_init_conf.mutable_constant_conf()->set_value(lamb_conf.beta2());
  KernelUtil<device_type, float>::InitializeWithProperConf(ctx, &beta1_init_conf, 0,
                                                           BnInOp2Blob("beta1_t"));
  KernelUtil<device_type, float>::InitializeWithProperConf(ctx, &beta2_init_conf, 0,
                                                           BnInOp2Blob("beta2_t"));
}

template<DeviceType device_type, typename T>
void LAMBMdUpdateKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, m_blob, "m",
                                                m_blob->shape().At(0), m_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, v_blob, "v",
                                                v_blob->shape().At(0), v_blob->shape().Count(1));
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  KernelUtil<device_type, float>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, beta1_t_blob, "beta1_t", beta1_t_blob->shape().At(0),
      beta1_t_blob->shape().Count(1));
  KernelUtil<device_type, float>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, beta2_t_blob, "beta2_t", beta2_t_blob->shape().At(0),
      beta2_t_blob->shape().Count(1));
}

template<DeviceType device_type, typename T>
void LAMBMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, const T* batch_instance_num_ptr, T learning_rate, T l1, T l2,
    int64_t next_model_vid, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* m_blob = BnInOp2Blob("m");
  Blob* v_blob = BnInOp2Blob("v");
  Blob* beta1_t_blob = BnInOp2Blob("beta1_t");
  Blob* beta2_t_blob = BnInOp2Blob("beta2_t");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  const LAMBModelUpdateConf& lamb_conf =
      this->op_conf().normal_mdupdt_conf().user_conf().lamb_conf();
  if (next_model_vid != 1) {
    KernelUtil<device_type, float>::Scal(ctx, 1, lamb_conf.beta1(), beta1_t_blob->mut_dptr<float>(),
                                         1);
    KernelUtil<device_type, float>::Scal(ctx, 1, lamb_conf.beta2(), beta2_t_blob->mut_dptr<float>(),
                                         1);
  }
  Memset<device_type>(ctx, fw_buf_blob->mut_dptr<T>(), 0,
                      fw_buf_blob->ByteSizeOfDataContentField());
  LAMBMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), batch_instance_num_ptr, learning_rate, l1, l2,
      static_cast<float>(lamb_conf.beta1()), static_cast<float>(lamb_conf.beta2()),
      static_cast<float>(lamb_conf.epsilon()), next_model_vid, beta1_t_blob->dptr<float>(),
      beta2_t_blob->dptr<float>(), BnInOp2Blob("model_diff")->mut_dptr<T>(),
      model_blob->mut_dptr<T>(), m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>(),
      fw_buf_blob->mut_dptr<T>());
}

template<typename T>
class LAMBMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const T* batch_instance_num_ptr,
                          T learning_rate, T l1, T l2, float beta1, float beta2, float epsilon,
                          int64_t next_model_vid, const float* beta1_t, const float* beta2_t,
                          T* model_diff, T* model, T* m, T* v, T* fw_buf) {
    FOR_RANGE(int64_t, i, 0, n) {
      model_diff[i] = model_diff[i] / *batch_instance_num_ptr;
      m[i] = beta1 * m[i] + (1 - beta1) * model_diff[i];
      v[i] = beta2 * v[i] + (1 - beta2) * (model_diff[i] * model_diff[i]);
      model_diff[i] =
          (m[i] / (1 - *beta1_t)) / std::sqrt(v[i] / (1 - *beta2_t) + epsilon) + l2 * model[i];
    }
    KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, model, 1, model, 1, &fw_buf[0]);
    KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, model_diff, 1, model_diff, 1, &fw_buf[1]);
    fw_buf[0] = std::sqrt(fw_buf[0]);
    fw_buf[1] = std::sqrt(fw_buf[1]);
    learning_rate = fw_buf[0] / fw_buf[1] * learning_rate;
    FOR_RANGE(int64_t, i, 0, n) { model[i] = model[i] - learning_rate * model_diff[i]; }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(LAMB);

}  // namespace oneflow
