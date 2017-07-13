#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

// momentum = beta * momentum - alpha * model_diff
// model = model - momentum
template<DeviceType device_type, typename FloatingPointType>
void MomentumMdUpdateKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* model_blob = BnInOp2BlobPtr("model");
  Blob* model_diffs_blob = BnInOp2BlobPtr("model_diffs");
  Blob* momentum_blob = BnInOp2BlobPtr("momentum");
  float learning_rate = op()->op_conf().momentum_model_update_conf().learning_rate();
  float beta = op()->op_conf().momentum_model_update_conf().beta();
  float alpha = learning_rate / JobDesc::Singleton()->batch_size();
  CHECK(std::isfinite(alpha));

  // momentum = beta * momentum
  KernelUtil<device_type, FloatingPointType>::BlasScal(
      ctx, momentum_blob->shape().elem_cnt(), beta,
      static_cast<FloatingPointType*>(momentum_blob->mut_dptr()), 1);

  // momentum = momentum - alpha * model_diff
  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
      ctx, momentum_blob->shape().elem_cnt(), -alpha,
      static_cast<const FloatingPointType*>(model_diffs_blob->dptr()), 1,
      static_cast<FloatingPointType*>(momentum_blob->mut_dptr()), 1);

  // model = model - momentum
  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
      ctx, model_blob->shape().elem_cnt(), -1,
      static_cast<const FloatingPointType*>(momentum_blob->dptr()), 1,
      static_cast<FloatingPointType*>(model_blob->mut_dptr()), 1);
}

INSTANTIATE_KERNEL_CLASS(MomentumMdUpdateKernel);
REGISTER_KERNEL(OperatorConf::kMomentumModelUpdateConf, MomentumMdUpdateKernel);

}  // namespace oneflow
