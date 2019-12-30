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
    NdarrayUtil<device_type, T>::BroadcastDiv(ctx.device_ctx, XpuVarNdarray<T>(out, num_axes),
                                              XpuVarNdarray<const T>(a, num_axes),
                                              XpuVarNdarray<const T>(b, num_axes));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastDivConf, BroadcastDivKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
