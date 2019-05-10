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
  NewKernelUtil<device_type>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans, OneVal<T>::value,
                                       ZeroVal<T>::value, in_blob, weight_blob, out_blob);

  if (this->op_conf().fully_connected_conf().use_bias()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");

    // out = bias_multiplier * bias + out
    NewKernelUtil<device_type>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
                                         OneVal<T>::value, OneVal<T>::value, bias_mul_blob,
                                         bias_blob, out_blob);
  }
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");

  Blob* in_diff_blob = BnInOp2Blob("in_diff");

  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");

  if (in_diff_blob != nullptr) {
    // in_diff = out_diff * weight
    NewKernelUtil<device_type>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
                                         OneVal<T>::value, ZeroVal<T>::value, out_diff_blob,
                                         weight_blob, in_diff_blob);
  }
  if (this->op_conf().trainable()) {
    // weight_diff = out_diff * in
    NewKernelUtil<device_type>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans, OneVal<T>::value,
                                         ZeroVal<T>::value, out_diff_blob, in_blob,
                                         weight_diff_blob);

    if (this->op_conf().fully_connected_conf().use_bias()) {
      const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
      Blob* bias_diff_blob = BnInOp2Blob("bias_diff");

      // bias_diff = bias_multiplier * out_diff
      NewKernelUtil<device_type>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                           OneVal<T>::value, ZeroVal<T>::value, bias_mul_blob,
                                           out_diff_blob, bias_diff_blob);
    }
  }
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
