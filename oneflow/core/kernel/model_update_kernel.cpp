#include "oneflow/core/kernel/model_update_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void MdUpdateKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* model_blob = BnInOp2BlobPtr("model");
  Blob* model_diffs_blob = BnInOp2BlobPtr("model_diffs");
  float learn_rate = op()->op_conf().model_update_conf().learn_rate();
  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
    ctx, model_blob->shape().elem_cnt(), -learn_rate,
    static_cast<const FloatingPointType*>(model_blob->dptr()), 1,
    static_cast<FloatingPointType*>(model_diffs->mut_dptr()), 1);
}

INSTANIATE_KERNEL_CALSS();
REGISTER_KERNEL();

}  // namespace oneflow
