#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* model_diff_acc_blob = BnInOp2Blob("model_diff_acc");
  Blob* model_blob = BnInOp2Blob("model");
  float learning_rate = this->op_conf().normal_mdupdt_conf().learning_rate();
  float alpha = learning_rate / JobDesc::Singleton()->BatchSize();
  CHECK(std::isfinite(alpha));

  // model = model - alpha * model_diff
  KernelUtil<device_type, T>::BlasAxpy(
      ctx.device_ctx, model_blob->shape().elem_cnt(), -alpha,
      model_diff_acc_blob->dptr<T>(), 1, model_blob->mut_dptr<T>(), 1);
}

namespace {

Kernel* CreateNormalMdUpdateKernel(DeviceType device_type,
                                   const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define MODEL_UPDATE_KERNEL_ENTRY(device_type, data_type_pair)            \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {     \
     return new NormalMdUpdateKernel<device_type,                         \
                                     OF_PP_PAIR_FIRST(data_type_pair)>(); \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
          MODEL_UPDATE_KERNEL_ENTRY, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(device_type, kernel_conf.data_type()))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kNormalMdupdtConf,
                         CreateNormalMdUpdateKernel))

}  // namespace oneflow
