#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

namespace {

const RMSPropModelUpdateConf& GetRMSPropModelUpdateConf(const OperatorConf& op_conf) {
  return op_conf.rmsprop_model_update_conf().user_conf().rmsprop_conf();
}

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& RMSPropMdUpdateKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().rmsprop_model_update_conf();
}

template<DeviceType device_type, typename T>
void RMSPropMdUpdateKernel<device_type, T>::UpdateModel(
    DeviceCtx* ctx, T weight_decay, const int64_t* train_step, const float* learning_rate,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_blob = BnInOp2Blob("model_diff");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* mean_square_blob = BnInOp2Blob("mean_square");
  const RMSPropModelUpdateConf& conf = GetRMSPropModelUpdateConf(this->op_conf());

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(), train_step, learning_rate,
      static_cast<T>(conf.decay_rate()), static_cast<T>(conf.epsilon()), weight_decay,
      model_diff_blob->dptr<T>(), model_blob->mut_dptr<T>(), mean_square_blob->mut_dptr<T>());
}

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(DeviceCtx*, int64_t n, const int64_t* train_step,
                          const float* learning_rate, T decay_rate, T epsilon, T weight_decay,
                          const T* model_diff, T* model, T* mean_square) {
    const T cur_decay_rate = *train_step == 0 ? 0 : decay_rate;
    for (int64_t i = 0; i < n; ++i) {
      T model_diff_val = model_diff[i];
      mean_square[i] =
          (1 - cur_decay_rate) * model_diff_val * model_diff_val + cur_decay_rate * mean_square[i];
      model[i] = model[i] - *learning_rate * model_diff_val / std::sqrt(mean_square[i] + epsilon);
    }
  }
};

DEFINE_MDUPDT_KERNEL_CREATOR(RMSProp);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRmspropModelUpdateConf, RMSPropMdUpdateKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
