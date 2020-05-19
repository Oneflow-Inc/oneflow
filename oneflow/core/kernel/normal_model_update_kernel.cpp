#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::VirtualKernelInit() {
  const PbMessage& op_conf = this->GetCustomizedOpConf();
  weight_decay_ = static_cast<T>(GetValFromPbMessage<float>(op_conf, "weight_decay"));
  if (!IsWeightDecaySupported()) { CHECK_EQ(weight_decay_, static_cast<T>(0)); }
}

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t* train_step_ptr = BnInOp2Blob("train_step")->dptr<int64_t>();
  const float* learning_rate_ptr = BnInOp2Blob("learning_rate")->dptr<float>();
  UpdateModel(ctx.device_ctx, weight_decay_, train_step_ptr, learning_rate_ptr, BnInOp2Blob);
}

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class NormalMdUpdateKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_KERNEL

}  // namespace oneflow
