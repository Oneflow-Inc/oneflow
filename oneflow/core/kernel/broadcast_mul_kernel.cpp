#include "oneflow/core/kernel/broadcast_mul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastMulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a = BnInOp2Blob("a");
  const Blob* b = BnInOp2Blob("b");
  Blob* out = BnInOp2Blob("out");
  int64_t n = out->shape().elem_cnt();
  if (a->shape().elem_cnt() == 1) {
    CHECK_EQ(n, b->shape().elem_cnt());
    KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, b->dptr<T>(), a->dptr<T>(),
                                            out->mut_dptr<T>());
  } else if (b->shape().elem_cnt() == 1) {
    CHECK_EQ(n, a->shape().elem_cnt());
    KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, a->dptr<T>(), b->dptr<T>(),
                                            out->mut_dptr<T>());
  } else {
    size_t num_axes = out->shape().NumAxes();
    XpuNdArrayUtil<device_type, T>::template Binary<BinaryFuncMul>::SwitchBroadcastApply(
        SwitchCase(num_axes), ctx.device_ctx, XpuVarNdarray<T>(out, num_axes),
        XpuVarNdarray<const T>(a, num_axes), XpuVarNdarray<const T>(b, num_axes));
  }
}

template<DeviceType device_type, typename T>
void BroadcastMulKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a = BnInOp2Blob("a");
  const Blob* b = BnInOp2Blob("b");
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* a_diff = BnInOp2Blob(GenDiffBn("a"));
  Blob* b_diff = BnInOp2Blob(GenDiffBn("b"));
  int64_t n = out_diff->shape().elem_cnt();
  if (a->shape().elem_cnt() == 1) {
    if (a_diff) {
      KernelUtil<device_type, T>::Dot(ctx.device_ctx, n, out_diff->dptr<T>(), 1, b->dptr<T>(), 1,
                                      a_diff->mut_dptr<T>());
    }
    if (b_diff) {
      KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, out_diff->dptr<T>(), a->dptr<T>(),
                                              b_diff->mut_dptr<T>());
    }
  } else if (b->shape().elem_cnt() == 1) {
    if (a_diff) {
      KernelUtil<device_type, T>::MulByScalar(ctx.device_ctx, n, out_diff->dptr<T>(), b->dptr<T>(),
                                              a_diff->mut_dptr<T>());
    }
    if (b_diff) {
      KernelUtil<device_type, T>::Dot(ctx.device_ctx, n, out_diff->dptr<T>(), 1, a->dptr<T>(), 1,
                                      b_diff->mut_dptr<T>());
    }
  } else {
    size_t num_axes = out_diff->shape().NumAxes();
    XpuVarNdarray<const T> out_diff_tensor(out_diff, num_axes);
    Blob* bw_buf_blob = BnInOp2Blob("bw_buf");
    XpuVarNdarray<const T> const_tmp(out_diff_tensor.shape(), bw_buf_blob->dptr<T>());
    XpuVarNdarray<T> tmp(out_diff_tensor.shape(), bw_buf_blob->mut_dptr<T>());
    if (a_diff) {
      XpuNdArrayUtil<device_type, T>::template Binary<BinaryFuncMul>::SwitchBroadcastApply(
          SwitchCase(num_axes), ctx.device_ctx, tmp, out_diff_tensor,
          XpuVarNdarray<const T>(b, num_axes));
      XpuNdArrayUtil<device_type, T>::SwitchReduce(
          SwitchCase(num_axes), ctx.device_ctx, XpuVarNdarray<T>(a_diff, num_axes), const_tmp, tmp);
    }
    if (b_diff) {
      XpuNdArrayUtil<device_type, T>::template Binary<BinaryFuncMul>::SwitchBroadcastApply(
          SwitchCase(num_axes), ctx.device_ctx, tmp, out_diff_tensor,
          XpuVarNdarray<const T>(a, num_axes));
      XpuNdArrayUtil<device_type, T>::SwitchReduce(
          SwitchCase(num_axes), ctx.device_ctx, XpuVarNdarray<T>(b_diff, num_axes), const_tmp, tmp);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastMulConf, BroadcastMulKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
