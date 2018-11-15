#include "oneflow/core/kernel/broadcast_mul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

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
    CHECK_EQ(n, a->shape().elem_cnt());
    CHECK(a->shape() == b->shape());
    KernelUtil<device_type, T>::Mul(ctx.device_ctx, n, a->dptr<T>(), b->dptr<T>(),
                                    out->mut_dptr<T>());
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
    CHECK(a->shape() == b->shape());
    if (a_diff) {
      KernelUtil<device_type, T>::Mul(ctx.device_ctx, n, out_diff->dptr<T>(), b->dptr<T>(),
                                      a_diff->mut_dptr<T>());
    }
    if (b_diff) {
      KernelUtil<device_type, T>::Mul(ctx.device_ctx, n, out_diff->dptr<T>(), a->dptr<T>(),
                                      b_diff->mut_dptr<T>());
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastMulConf, BroadcastMulKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
