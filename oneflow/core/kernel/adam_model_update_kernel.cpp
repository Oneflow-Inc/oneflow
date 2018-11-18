#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
void UpdateMomentEstimate(int64_t n, T beta, int32_t p, const T* model_diff, const T* beta_t,
                          T* momentum) {
  FOR_RANGE(int64_t, i, 0, n) {
    // Update biased moment estimate
    momentum[i] = beta * momentum[i] + (1 - beta) * std::pow(model_diff[i], p);
    // Correct deviation of moment estimate
    momentum[i] = momentum[i] / (1 - *beta_t);
  }
}

}  // namespace

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
  if (next_model_vid == 1) {
    Memset<device_type>(ctx, m_blob->mut_dptr<T>(), 0, m_blob->ByteSizeOfDataContentField());
    Memset<device_type>(ctx, v_blob->mut_dptr<T>(), 0, v_blob->ByteSizeOfDataContentField());
  } else {
    KernelUtil<device_type, T>::Axpy(ctx, 1, static_cast<T>(adam_conf.beta1()),
                                     beta1_t_blob->dptr<T>(), 1, beta1_t_blob->mut_dptr<T>(), 1);
    KernelUtil<device_type, T>::Axpy(ctx, 1, static_cast<T>(adam_conf.beta2()),
                                     beta2_t_blob->dptr<T>(), 1, beta2_t_blob->mut_dptr<T>(), 1);
  }
  AdamMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), batch_instance_num_ptr, learning_rate, l1, l2,
      static_cast<T>(adam_conf.beta1()), static_cast<T>(adam_conf.beta2()),
      static_cast<T>(adam_conf.epsilon()), next_model_vid, beta1_t_blob->dptr<T>(),
      beta2_t_blob->dptr<T>(), BnInOp2Blob("model_diff")->mut_dptr<T>(), model_blob->mut_dptr<T>(),
      m_blob->mut_dptr<T>(), v_blob->mut_dptr<T>());
}

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const T* batch_instance_num_ptr,
                          T learning_rate, T l1, T l2, T beta1, T beta2, T epsilon,
                          int64_t next_model_vid, const T* beta1_t, const T* beta2_t, T* model_diff,
                          T* model, T* m, T* v) {
    // first-order moment
    UpdateMomentEstimate<T>(n, beta1, 1, model_diff, beta1_t, m);
    // second-order moment
    UpdateMomentEstimate<T>(n, beta2, 2, model_diff, beta2_t, v);
    FOR_RANGE(int64_t, i, 0, n) {
      model_diff[i] = m[i] / (std::sqrt(v[i]) + epsilon);
      T reg_diff = RegularizeDiff(model_diff[i], *batch_instance_num_ptr, l1, l2, model[i]);
      model[i] = model[i] - learning_rate * reg_diff;
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Adam);

}  // namespace oneflow
