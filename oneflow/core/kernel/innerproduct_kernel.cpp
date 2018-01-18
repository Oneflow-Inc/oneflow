#include "oneflow/core/kernel/innerproduct_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");

  // out = in * weight
  KernelUtil<device_type, T>::BlasMatrixMatrix(
      ctx, CblasNoTrans, CblasTrans, static_cast<T>(1.0), static_cast<T>(0.0),
      in_blob, weight_blob, out_blob);

  if (this->op_conf().innerproduct_conf().has_bias_term()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");

    // out = bias_multiplier * bias + out
    KernelUtil<device_type, T>::BlasMatrixMatrix(
        ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1.0),
        static_cast<T>(1.0), bias_mul_blob, bias_blob, out_blob);
  }
}

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");

  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");

  // weight_diff = out_diff * in
  KernelUtil<device_type, T>::BlasMatrixMatrix(
      ctx, CblasTrans, CblasNoTrans, static_cast<T>(1.0), static_cast<T>(0.0),
      out_diff_blob, in_blob, weight_diff_blob);

  // in_diff = out_diff * weight
  if (in_diff_blob != nullptr) {
    KernelUtil<device_type, T>::BlasMatrixMatrix(
        ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1.0),
        static_cast<T>(0.0), out_diff_blob, weight_blob, in_diff_blob);
  }

  if (this->op_conf().innerproduct_conf().has_bias_term()) {
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");

    // bias_diff = bias_multiplier * out_diff
    KernelUtil<device_type, T>::BlasMatrixMatrix(
        ctx, CblasTrans, CblasNoTrans, static_cast<T>(1.0), static_cast<T>(0.0),
        bias_mul_blob, out_diff_blob, bias_diff_blob);
  }
}

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().innerproduct_conf(),
                        weight_initializer),
      random_seed_gen(), BnInOp2Blob("weight"));

  if (this->op_conf().innerproduct_conf().has_bias_term()) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(this->op_conf().innerproduct_conf(),
                          bias_initializer),
        random_seed_gen(), BnInOp2Blob("bias"));
  }
}
template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = this->op_conf().innerproduct_conf().out_num();
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, weight_blob, "weight",
      dim_num, weight_blob->shape().Count(1));
  if (this->op_conf().innerproduct_conf().has_bias_term()) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"),
        "bias", dim_num, 1);
  }
}

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->op_conf().innerproduct_conf().has_bias_term()) {
    InitializerConf bias_multiplier_initializer_conf;
    bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::Initialize(ctx.device_ctx,
                                           bias_multiplier_initializer_conf, 0,
                                           BnInOp2Blob("bias_multiplier"));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kInnerproductConf, InnerProductKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
