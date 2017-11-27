#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RMSPropMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_acc_blob = BnInOp2Blob("model_diff_acc");
  Blob* model_blob = BnInOp2Blob("model");
  Blob* mean_square_blob = BnInOp2Blob("mean_square");
  const RMSPropModelUpdateOpConf& conf = this->op_conf().rmsprop_mdupdt_conf();
  const float batch_size = JobDesc::Singleton()->BatchSize();
  float decay_rate = conf.decay_rate();
  if (*reinterpret_cast<int64_t*>(ctx.other) == 1) { decay_rate = 0.0f; }

  RMSPropMdUpdateKernelUtil<device_type, T>::UpdateModel(
      ctx, model_blob->shape().elem_cnt(),
      static_cast<T>((1.0f - decay_rate) / (batch_size * batch_size)),
      static_cast<T>(conf.learning_rate() / batch_size),
      static_cast<T>(decay_rate), static_cast<T>(conf.epsilon()),
      model_blob->mut_dptr<T>(), mean_square_blob->mut_dptr<T>(),
      model_diff_acc_blob->dptr<T>());
}

template<typename T>
class RMSPropMdUpdateKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void UpdateModel(const KernelCtx& ctx, const int64_t n, const T alpha,
                          const T learning_rate, const T decay_rate,
                          const T epsilon, T* model, T* mean_square,
                          const T* model_diff_acc) {
    for (int64_t i = 0; i < n; ++i) {
      mean_square[i] = alpha * model_diff_acc[i] * model_diff_acc[i]
                       + decay_rate * mean_square[i];
      model[i] -= learning_rate * model_diff_acc[i]
                  / (std::sqrt(mean_square[i] + epsilon));
    }
  }
};

namespace {

Kernel* CreateRMSPropMdUpdateKernel(DeviceType device_type,
                                    const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define MODEL_UPDATE_KERNEL_ENTRY(device_type, data_type_pair)             \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {      \
     return new RMSPropMdUpdateKernel<device_type,                         \
                                      OF_PP_PAIR_FIRST(data_type_pair)>(); \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
          MODEL_UPDATE_KERNEL_ENTRY, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(device_type, kernel_conf.data_type()))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kRmspropMdupdtConf,
                         CreateRMSPropMdUpdateKernel))

}  // namespace oneflow
