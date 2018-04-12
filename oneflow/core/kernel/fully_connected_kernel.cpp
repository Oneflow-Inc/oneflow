#include "oneflow/core/kernel/fully_connected_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");

  // out = in * weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       OneVal<T>::value, ZeroVal<T>::value,
                                       in_blob, weight_blob, out_blob);

  if (this->op_conf().fully_connected_conf().use_bias()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");

    // out = bias_multiplier * bias + out
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, OneVal<T>::value,
        OneVal<T>::value, bias_mul_blob, bias_blob, out_blob);
  }
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = this->GetOutDiffBlob(BnInOp2Blob);

  Blob* in_diff_blob = BnInOp2Blob("in_diff");

  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");

  // weight_diff = out_diff * in
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasTrans, CblasNoTrans, OneVal<T>::value,
      ZeroVal<T>::value, out_diff_blob, in_blob, weight_diff_blob);

  // in_diff = out_diff * weight
  if (in_diff_blob != nullptr) {
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, OneVal<T>::value,
        ZeroVal<T>::value, out_diff_blob, weight_blob, in_diff_blob);
  }

  if (this->op_conf().fully_connected_conf().use_bias()) {
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");

    // bias_diff = bias_multiplier * out_diff
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasTrans, CblasNoTrans, OneVal<T>::value,
        ZeroVal<T>::value, bias_mul_blob, out_diff_blob, bias_diff_blob);
  }
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (!this->op_conf().fully_connected_conf().use_bias()) { return; }
  InitializerConf bias_multiplier_initializer_conf;
  bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  KernelUtil<device_type, T>::InitializeWithConf(
      ctx, bias_multiplier_initializer_conf, 0, BnInOp2Blob("bias_multiplier"));
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      GetMsgPtrFromPbMessage(this->op_conf().fully_connected_conf(),
                             "weight_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("weight"));
  if (this->op_conf().fully_connected_conf().use_bias()) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx,
        GetMsgPtrFromPbMessage(this->op_conf().fully_connected_conf(),
                               "bias_initializer"),
        (*random_seed_gen)(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = this->op_conf().fully_connected_conf().units();
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, weight_blob, "weight", dim_num,
      weight_blob->shape().Count(1));
  if (this->op_conf().fully_connected_conf().use_bias()) {
    KernelUtil<device_type, T>::InitializeWithDir(
        ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"), "bias",
        dim_num, 1);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& FullyConnectedKernel<device_type, T>::GetCustomizedOpConf()
    const {
  return this->op_conf().fully_connected_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFullyConnectedConf,
                           FullyConnectedKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
