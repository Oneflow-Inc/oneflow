#include "oneflow/core/kernel/bias_add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BiasAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a_blob = BnInOp2Blob("a");
  const Blob* b_blob = BnInOp2Blob("b");
  const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
  Blob* out_blob = BnInOp2Blob("out");

  // out = bias_multiplier * b + a
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), a_blob->dptr<T>(),
                      a_blob->ByteSizeOfDataContentField());
  KernelUtil<device_type, T>::OFGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
                                     out_blob->shape().At(0), out_blob->shape().At(1), 1,
                                     OneVal<T>::value, bias_mul_blob->dptr<T>(), b_blob->dptr<T>(),
                                     OneVal<T>::value, out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BiasAddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
  Blob* a_diff_blob = BnInOp2Blob("a_diff");
  Blob* b_diff_blob = BnInOp2Blob("b_diff");

  Memcpy<device_type>(ctx.device_ctx, a_diff_blob->mut_dptr<T>(), out_diff_blob->dptr<T>(),
                      out_diff_blob->ByteSizeOfDataContentField());
  // b_diff = bias_multiplier * out_diff
  KernelUtil<device_type, T>::OFGemm(
      ctx.device_ctx, CblasTrans, CblasNoTrans, 1, b_diff_blob->shape().At(0),
      out_diff_blob->shape().At(0), OneVal<T>::value, bias_mul_blob->dptr<T>(),
      out_diff_blob->dptr<T>(), ZeroVal<T>::value, b_diff_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BiasAddKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf bias_multiplier_initializer_conf;
  bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  KernelUtil<device_type, T>::InitializeWithConf(ctx, bias_multiplier_initializer_conf, 0,
                                                 BnInOp2Blob("bias_multiplier"));
}

template<DeviceType device_type, typename T>
const PbMessage& BiasAddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().bias_add_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBiasAddConf, BiasAddKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
