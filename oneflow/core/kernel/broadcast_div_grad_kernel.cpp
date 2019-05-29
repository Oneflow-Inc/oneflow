#include "oneflow/core/kernel/broadcast_div_grad_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastDivGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* b_blob = BnInOp2Blob("b");
  const Blob* y_blob = BnInOp2Blob("y");
  const Blob* dy_blob = BnInOp2Blob("dy");
  Blob* tmp_blob = BnInOp2Blob("temp_storage");
  Blob* db_blob = BnInOp2Blob("db");

  const int64_t num_axes = dy_blob->shape().NumAxes();
  XpuVarNdarray<const T> dy(dy_blob, num_axes);
  XpuVarNdarray<const T> const_tmp(dy.shape(), tmp_blob->dptr<T>());
  XpuVarNdarray<T> tmp(dy.shape(), tmp_blob->mut_dptr<T>());

  NdarrayUtil<device_type, T>::template BroadcastApply<BinaryFuncDiv>(
      ctx.device_ctx, tmp, XpuVarNdarray<const T>(y_blob, num_axes),
      XpuVarNdarray<const T>(b_blob, num_axes));
  NdarrayUtil<device_type, T>::template BroadcastApply<BinaryFuncMul>(ctx.device_ctx, tmp, dy,
                                                                      const_tmp);
  NdarrayUtil<device_type, T>::ReduceSum(ctx.device_ctx, XpuVarNdarray<T>(db_blob, num_axes),
                                         const_tmp, tmp);
  NdarrayUtil<device_type, T>::template ImplaceApplyUnary<UnaryFuncNegative>(
      ctx.device_ctx, XpuVarNdarray<T>(db_blob, num_axes));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastDivGradConf, BroadcastDivGradKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
