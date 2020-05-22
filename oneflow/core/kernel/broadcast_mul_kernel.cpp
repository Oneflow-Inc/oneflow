#include "oneflow/core/kernel/broadcast_mul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

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
    NdarrayUtil<device_type, T>::BroadcastMul(ctx.device_ctx, XpuVarNdarray<T>(out, num_axes),
                                              XpuVarNdarray<const T>(a, num_axes),
                                              XpuVarNdarray<const T>(b, num_axes));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastMulConf, BroadcastMulKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace

}  // namespace oneflow
