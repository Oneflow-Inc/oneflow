#include "oneflow/core/kernel/square_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SquareKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<device_type, T>::Square(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                     in_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void SquareKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                  out_diff_blob->dptr<T>(), in_blob->dptr<T>(),
                                  in_diff_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                   static_cast<T>(2), in_diff_blob->mut_dptr<T>(), 1);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSquareConf, SquareKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
