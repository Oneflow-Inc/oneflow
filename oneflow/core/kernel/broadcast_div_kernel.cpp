#include "oneflow/core/kernel/broadcast_div_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastDivKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a = BnInOp2Blob("a");
  const Blob* b = BnInOp2Blob("b");
  Blob* out = BnInOp2Blob("out");
  int64_t n = out->shape().elem_cnt();
  if (a->shape().elem_cnt() == 1) {
    CHECK_EQ(n, b->shape().elem_cnt());
    KernelUtil<device_type, T>::Replicate(ctx.device_ctx, n, out->mut_dptr<T>(), a->dptr<T>());
    KernelUtil<device_type, T>::Div(ctx.device_ctx, n, out->dptr<T>(), b->dptr<T>(),
                                    out->mut_dptr<T>());
  } else if (b->shape().elem_cnt() == 1) {
    CHECK_EQ(n, a->shape().elem_cnt());
    KernelUtil<device_type, T>::Replicate(ctx.device_ctx, n, out->mut_dptr<T>(), b->dptr<T>());
    KernelUtil<device_type, T>::Div(ctx.device_ctx, n, a->dptr<T>(), out->dptr<T>(),
                                    out->mut_dptr<T>());
  } else {
    size_t num_axes = out->shape().NumAxes();
    NdarrayUtil<device_type, T>::template Binary<BinaryFuncDiv>::SwitchBroadcastApply(
        SwitchCase(num_axes), ctx.device_ctx, XpuVarNdarray<T>(out, num_axes),
        XpuVarNdarray<const T>(a, num_axes), XpuVarNdarray<const T>(b, num_axes));
  }
}

template<DeviceType device_type, typename T>
void BroadcastDivKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a = BnInOp2Blob("a");
  const Blob* b = BnInOp2Blob("b");
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* tmp = BnInOp2Blob("bw_buf");
  Blob* a_diff = BnInOp2Blob(GenDiffBn("a"));
  Blob* b_diff = BnInOp2Blob(GenDiffBn("b"));

  int64_t n = out_diff->shape().elem_cnt();
  KernelUtil<device_type, T>::Reciprocal(ctx.device_ctx, b->shape().elem_cnt(), b->dptr<T>(),
                                         tmp->mut_dptr<T>());
  if (a->shape().elem_cnt() == 1) {
    if (a_diff) {
      KernelUtil<device_type, T>::Dot(ctx.device_ctx, n, out_diff->dptr<T>(), 1, tmp->dptr<T>(), 1,
                                      a_diff->mut_dptr<T>());
    }
    if (b_diff) {
      KernelUtil<device_type, T>::Square(ctx.device_ctx, n, tmp->dptr<T>(), tmp->mut_dptr<T>());
      KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, tmp->dptr<T>(), a->dptr<T>(),
                                              tmp->mut_dptr<T>());
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, n, static_cast<T>(-2), tmp->dptr<T>(), 1,
                                       tmp->mut_dptr<T>(), 1);
      KernelUtil<device_type, T>::Mul(ctx.device_ctx, n, out_diff->dptr<T>(), tmp->dptr<T>(),
                                      b_diff->mut_dptr<T>());
    }
  } else if (b->shape().elem_cnt() == 1) {
    if (a_diff) {
      KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, out_diff->dptr<T>(),
                                              tmp->dptr<T>(), a_diff->mut_dptr<T>());
    }
    if (b_diff) {
      KernelUtil<device_type, T>::Square(ctx.device_ctx, 1, tmp->dptr<T>(), tmp->mut_dptr<T>());
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, 1, static_cast<T>(-2), tmp->dptr<T>(), 1,
                                       tmp->mut_dptr<T>(), 1);
      KernelUtil<device_type, T>::Dot(ctx.device_ctx, n, out_diff->dptr<T>(), 1, a->dptr<T>(), 1,
                                      b_diff->mut_dptr<T>());
      KernelUtil<device_type, T>::Mul(ctx.device_ctx, 1, tmp->dptr<T>(), b_diff->dptr<T>(),
                                      b_diff->mut_dptr<T>());
    }
  } else {
    size_t num_axes = out_diff->shape().NumAxes();
    XpuVarNdarray<const T> out_diff_tensor(out_diff, num_axes);
    Blob* bw_buf_blob = BnInOp2Blob("bw_buf");
    XpuVarNdarray<const T> const_tmp(out_diff_tensor.shape(), bw_buf_blob->dptr<T>());
    XpuVarNdarray<T> tmp(out_diff_tensor.shape(), bw_buf_blob->mut_dptr<T>());
    if (a_diff) {
      NdarrayUtil<device_type, T>::template Binary<BinaryFuncDiv>::SwitchBroadcastApply(
          SwitchCase(num_axes), ctx.device_ctx, tmp, out_diff_tensor,
          XpuVarNdarray<const T>(b, num_axes));
      NdarrayUtil<device_type, T>::SwitchReduce(SwitchCase(num_axes), ctx.device_ctx,
                                                XpuVarNdarray<T>(a_diff, num_axes), const_tmp, tmp);
    }
    if (b_diff) {
      const Blob* out_blob = BnInOp2Blob("out");
      NdarrayUtil<device_type, T>::template Binary<BinaryFuncDiv>::SwitchBroadcastApply(
          SwitchCase(num_axes), ctx.device_ctx, tmp, XpuVarNdarray<const T>(out_blob, num_axes),
          XpuVarNdarray<const T>(b, num_axes));
      NdarrayUtil<device_type, T>::template Binary<BinaryFuncMul>::SwitchBroadcastApply(
          SwitchCase(num_axes), ctx.device_ctx, tmp, out_diff_tensor, const_tmp);
      NdarrayUtil<device_type, T>::SwitchReduce(SwitchCase(num_axes), ctx.device_ctx,
                                                XpuVarNdarray<T>(b_diff, num_axes), const_tmp, tmp);
      NdarrayUtil<device_type, T>::template ImplaceApplyUnary<UnaryFuncMinus>(
          ctx.device_ctx, XpuVarNdarray<T>(b_diff, num_axes));
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastDivConf, BroadcastDivKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
