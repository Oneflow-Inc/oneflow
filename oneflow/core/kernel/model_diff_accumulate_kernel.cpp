#include "oneflow/core/kernel/model_diff_accumulate_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void MdDiffAccKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob = BnInOp2BlobPtr("model_diff");
  Blob* out_blob = BnInOp2BlobPtr("model_diff_acc");
  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
      ctx, in_blob->shape().elem_cnt(), static_cast<FloatingPointType>(1.0),
      static_cast<const FloatingPointType*>(in_blob->dptr()), 1,
      static_cast<FloatingPointType*>(out_blob->mut_dptr()), 1);
}

INSTANTIATE_KERNEL_CLASS(MdDiffAccKernel);
REGISTER_KERNEL(OperatorConf::kModelDiffAccConf, MdDiffAccKernel);

}  // namespace oneflow
