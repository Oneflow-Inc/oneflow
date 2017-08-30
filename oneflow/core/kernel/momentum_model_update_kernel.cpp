#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MomentumMdUpdateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* model_blob = BnInOp2Blob("model");
  Blob* momentum_blob = BnInOp2Blob("momentum");
  const Blob* model_diffs_blob = BnInOp2Blob("model_diffs");
  float learning_rate = op()->op_conf().momentum_mdupdt_conf().learning_rate();
  float beta = op()->op_conf().momentum_mdupdt_conf().beta();
  float alpha = learning_rate / JobDesc::Singleton()->batch_size();
  CHECK(std::isfinite(alpha));

  // momentum = beta * momentum
  KernelUtil<device_type, T>::BlasScal(
      ctx.device_ctx, momentum_blob->shape().elem_cnt(), static_cast<T>(beta),
      momentum_blob->mut_dptr<T>(), 1);

  // momentum = momentum - alpha * model_diff
  KernelUtil<device_type, T>::BlasAxpy(
      ctx.device_ctx, momentum_blob->shape().elem_cnt(), static_cast<T>(-alpha),
      model_diffs_blob->dptr<T>(), 1, momentum_blob->mut_dptr<T>(), 1);

  // model = model + momentum
  KernelUtil<device_type, T>::BlasAxpy(
      ctx.device_ctx, model_blob->shape().elem_cnt(), static_cast<T>(1),
      momentum_blob->dptr<T>(), 1, model_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void MomentumMdUpdateKernel<device_type, T>::InitDataTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FillConf momentum_fill_conf;
  momentum_fill_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, T>::Fill(ctx.device_ctx, momentum_fill_conf, 0,
                                   BnInOp2Blob("momentum"));
}

namespace {

template<DeviceType device_type>
Kernel* CreateMomentumMdUpdateKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define MOMENTUM_MDUPDATE_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto,                                               \
   []() { return new MomentumMdUpdateKernel<device_type, type_cpp>; }},
      OF_PP_FOR_EACH_TUPLE(MOMENTUM_MDUPDATE_KERNEL_ENTRY,
                           FLOATING_DATA_TYPE_SEQ)};
  return data_type2creator.at(JobDesc::Singleton()->default_data_type())();
}

}  // namespace

REGISTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kMomentumMdupdtConf,
                                 CreateMomentumMdUpdateKernel);

}  // namespace oneflow
