#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void MdUpdateKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* model_blob = BnInOp2BlobPtr("model");
  Blob* model_diffs_blob = BnInOp2BlobPtr("model_diffs");
  float learning_rate = op()->op_conf().model_update_conf().learning_rate();
  float alpha = -learning_rate / JobDesc::Singleton()->total_batch_num();
  CHECK(std::isfinite(alpha));

  // model = model - alpha * model_diff
  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
      ctx, model_blob->shape().elem_cnt(), alpha,
      static_cast<const FloatingPointType*>(model_diffs_blob->dptr()), 1,
      static_cast<FloatingPointType*>(model_blob->mut_dptr()), 1);
}

INSTANTIATE_KERNEL_CLASS(MdUpdateKernel);
REGISTER_KERNEL(OperatorConf::kModelUpdateConf, MdUpdateKernel);

}  // namespace oneflow
