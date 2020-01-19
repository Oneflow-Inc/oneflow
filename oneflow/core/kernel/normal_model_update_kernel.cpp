#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/naive_model_update_kernel.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"
#include "oneflow/core/kernel/lars_model_update_kernel.h"
#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/lazy_adam_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::VirtualKernelInit() {
  const PbMessage& op_conf = this->GetCustomizedOpConf();
  l1_ = static_cast<T>(GetValFromPbMessage<float>(op_conf, "l1"));
  l2_ = static_cast<T>(GetValFromPbMessage<float>(op_conf, "l2"));
}

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t* train_step_ptr = BnInOp2Blob("train_step")->dptr<int64_t>();
  const float* learning_rate_ptr = BnInOp2Blob("learning_rate")->dptr<float>();
  UpdateModel(ctx.device_ctx, l1_, l2_, train_step_ptr, learning_rate_ptr, BnInOp2Blob);
}

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template struct NormalMdUpdateKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
