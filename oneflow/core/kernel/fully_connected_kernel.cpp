#include "oneflow/core/kernel/fully_connected_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");

  // out = in * weight
  NewKernelUtil<device_type>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans, GetOneVal<T>(),
                                       GetZeroVal<T>(), in_blob, weight_blob, out_blob);

  if (this->op_conf().fully_connected_conf().use_bias()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");

    // out = bias_multiplier * bias + out
    NewKernelUtil<device_type>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans, GetOneVal<T>(),
                                         GetOneVal<T>(), bias_mul_blob, bias_blob, out_blob);
  }
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (!this->op_conf().fully_connected_conf().use_bias()) { return; }
  InitializerConf bias_multiplier_initializer_conf;
  bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  NewKernelUtil<device_type>::InitializeWithConstConf(
      ctx, bias_multiplier_initializer_conf.constant_conf(), BnInOp2Blob("bias_multiplier"));
}

template<DeviceType device_type, typename T>
const PbMessage& FullyConnectedKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().fully_connected_conf();
}

namespace {

Kernel* CreateFcKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (FullyConnectedKernel),
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)
          MAKE_KERNEL_CREATOR_ENTRY(FullyConnectedKernel, DeviceType::kGPU,
                                    (float16, DataType::kFloat16))};

  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), kernel_conf.data_type()))();
}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kFullyConnectedConf, CreateFcKernel);

}  // namespace

}  // namespace oneflow
