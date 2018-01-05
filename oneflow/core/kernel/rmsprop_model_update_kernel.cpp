#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RMSPropMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, const Blob* pre_model_blob, const Blob* model_diff_blob,
    int64_t next_model_vid,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* mean_square_blob = BnInOp2Blob("mean_square");
  const RMSPropModelUpdateOpConf& conf = this->op_conf().rmsprop_mdupdt_conf();
  float decay_rate = conf.decay_rate();
  if (next_model_vid == 1) { decay_rate = 0.0f; }

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), static_cast<T>(1.0f - decay_rate),
      static_cast<T>(conf.learning_rate()), static_cast<T>(decay_rate),
      static_cast<T>(conf.epsilon()), pre_model_blob->dptr<T>(),
      model_blob->mut_dptr<T>(), mean_square_blob->mut_dptr<T>(),
      model_diff_blob->dptr<T>());
}

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, const int64_t n, const T alpha,
                          const T learning_rate, const T decay_rate,
                          const T epsilon, const T* pre_model, T* model,
                          T* mean_square, const T* model_diff) {
    for (int64_t i = 0; i < n; ++i) {
      mean_square[i] =
          alpha * model_diff[i] * model_diff[i] + decay_rate * mean_square[i];
      model[i] =
          pre_model[i]
          - learning_rate * model_diff[i] / std::sqrt(mean_square[i] + epsilon);
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRmspropMdupdtConf,
                           RMSPropMdUpdateKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
