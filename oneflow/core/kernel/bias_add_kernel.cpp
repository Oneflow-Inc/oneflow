#include "oneflow/core/kernel/bias_add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

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
  NewKernelUtil<device_type>::OFGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
                                     out_blob->shape().At(0), out_blob->shape().At(1), 1,
                                     OneVal<T>::value, bias_mul_blob->dptr<T>(), b_blob->dptr<T>(),
                                     OneVal<T>::value, out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
const PbMessage& BiasAddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().bias_add_conf();
}

namespace {

Kernel* CreateBiasAddKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (BiasAddKernel), DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ)
          MAKE_KERNEL_CREATOR_ENTRY(BiasAddKernel, DeviceType::kGPU,
                                    (float16, DataType::kFloat16))};

  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), kernel_conf.data_type()))();
}

REGISTER_KERNEL_CREATOR(OperatorConf::kBiasAddConf, CreateBiasAddKernel);

}  // namespace

}  // namespace oneflow
