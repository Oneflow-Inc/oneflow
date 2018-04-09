#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MomentumMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, const Blob* pre_model_blob, const Blob* model_diff_blob,
    int64_t next_model_vid, double learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  float beta =
      this->op_conf().normal_mdupdt_conf().user_conf().momentum_conf().beta();
  if (next_model_vid == 1) { beta = 0.0f; }

  MomentumMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), static_cast<T>(beta),
      static_cast<T>(learning_rate), model_diff_blob->dptr<T>(),
      pre_model_blob->dptr<T>(), momentum_blob->mut_dptr<T>(),
      model_blob->mut_dptr<T>());
}

template<typename T>
class MomentumMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, const int64_t n, const T beta,
                          const T learning_rate, const T* model_diff,
                          const T* pre_model, T* momentum, T* model) {
    for (int64_t i = 0; i != n; ++i) {
      momentum[i] = beta * momentum[i] - learning_rate * model_diff[i];
      model[i] = pre_model[i] + momentum[i];
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(Momentum);

}  // namespace oneflow
