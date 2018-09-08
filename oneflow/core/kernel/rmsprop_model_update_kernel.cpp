#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

template<DeviceType device_type, typename T>
void RMSPropMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, int64_t batch_size, T learning_rate, T l1, T l2, const Blob* pre_model_blob,
    int64_t next_model_vid, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* mean_square_blob = BnInOp2Blob("mean_square");
  const RMSPropModelUpdateConf& conf =
      this->op_conf().normal_mdupdt_conf().user_conf().rmsprop_conf();
  float decay_rate = conf.decay_rate();
  if (next_model_vid == 1) { decay_rate = 0.0f; }

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), batch_size, learning_rate, static_cast<T>(decay_rate),
      static_cast<T>(conf.epsilon()), l1, l2, pre_model_blob->dptr<T>(), model_blob->mut_dptr<T>(),
      mean_square_blob->mut_dptr<T>(), model_diff_blob->dptr<T>());
}

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, int64_t batch_size, T learning_rate, T decay_rate,
                          T epsilon, T l1, T l2, const T* pre_model, T* model, T* mean_square,
                          const T* model_diff) {
    for (int64_t i = 0; i < n; ++i) {
      T reg_diff = RegularizeDiff(model_diff[i], batch_size, l1, l2, pre_model[i]);
      mean_square[i] = (1 - decay_rate) * reg_diff * reg_diff + decay_rate * mean_square[i];
      model[i] = pre_model[i] - learning_rate * reg_diff / std::sqrt(mean_square[i] + epsilon);
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(RMSProp);

}  // namespace oneflow
