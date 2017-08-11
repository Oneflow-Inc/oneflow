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
  float learning_rate = op()->op_conf().momentum_mdupdt_conf().learning_rate();
  float beta = op()->op_conf().momentum_mdupdt_conf().beta();
  float alpha = learning_rate / JobDesc::Singleton()->batch_size();
  CHECK(std::isfinite(alpha));

  // momentum = beta * momentum
  KernelUtil<device_type, FloatingPointType>::BlasScal(
      ctx, momentum_blob->shape().elem_cnt(),
      static_cast<FloatingPointType>(beta),
      momentum_blob->mut_dptr<FloatingPointType>(), 1);

  // momentum = momentum - alpha * model_diff
  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
      ctx, momentum_blob->shape().elem_cnt(),
      static_cast<FloatingPointType>(-alpha),
      model_diffs_blob->dptr<FloatingPointType>(), 1,
      momentum_blob->mut_dptr<FloatingPointType>(), 1);

  // model = model + momentum
  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
      ctx, model_blob->shape().elem_cnt(), static_cast<FloatingPointType>(1),
      momentum_blob->dptr<FloatingPointType>(), 1,
      model_blob->mut_dptr<FloatingPointType>(), 1);
}

template<DeviceType device_type, typename FloatingPointType>
void MomentumMdUpdateKernel<device_type, FloatingPointType>::InitDataTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* momentum_blob = BnInOp2Blob("momentum");
  FloatingPointType* dptr = momentum_blob->mut_dptr<FloatingPointType>();
  std::fill(dptr, dptr + momentum_blob->shape().elem_cnt(),
            static_cast<FloatingPointType>(0));
}

INSTANTIATE_KERNEL_CLASS(MomentumMdUpdateKernel);
REGISTER_KERNEL(OperatorConf::kMomentumMdupdtConf, MomentumMdUpdateKernel);

}  // namespace oneflow
