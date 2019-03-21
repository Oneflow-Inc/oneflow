#include "oneflow/core/kernel/broadcast_div_grad_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastDivGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* b = BnInOp2Blob("b");
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  const Blob* out_blob = BnInOp2Blob("out");
  Blob* tmp_blob = BnInOp2Blob("temp_storage");
  Blob* b_diff = BnInOp2Blob(GenDiffBn("b"));

  KernelUtil<device_type, T>::Reciprocal(ctx.device_ctx, b->shape().elem_cnt(), b->dptr<T>(),
                                         tmp_blob->mut_dptr<T>());

  const int64_t num_axes = out_diff->shape().NumAxes();
  XpuVarNdarray<const T> out_diff_tensor(out_diff, num_axes);
  XpuVarNdarray<const T> const_tmp(out_diff_tensor.shape(), tmp_blob->dptr<T>());
  XpuVarNdarray<T> tmp(out_diff_tensor.shape(), tmp_blob->mut_dptr<T>());


  NdarrayUtil<device_type, T>::template BroadcastApply<BinaryFuncDiv>(
      ctx.device_ctx, tmp, XpuVarNdarray<const T>(out_blob, num_axes),
      XpuVarNdarray<const T>(b, num_axes));
  NdarrayUtil<device_type, T>::template BroadcastApply<BinaryFuncMul>(
      ctx.device_ctx, tmp, out_diff_tensor, const_tmp);
  NdarrayUtil<device_type, T>::ReduceSum(ctx.device_ctx, XpuVarNdarray<T>(b_diff, num_axes),
                                         const_tmp, tmp);
  NdarrayUtil<device_type, T>::template ImplaceApplyUnary<UnaryFuncMinus>(
      ctx.device_ctx, XpuVarNdarray<T>(b_diff, num_axes));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDenominatorGradConf, BroadcastDivGradKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
