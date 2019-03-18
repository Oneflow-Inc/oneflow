#include "oneflow/core/kernel/broadcast_like_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastLikeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x = BnInOp2Blob("x");
  const Blob* like = BnInOp2Blob("like");
  Blob* y = BnInOp2Blob("y");
  int64_t n = y->shape().elem_cnt();
  if (x->shape().elem_cnt() == 1) {
    CHECK_EQ(n, like->shape().elem_cnt());
    KernelUtil<device_type, T>::Replicate(ctx.device_ctx, n, y->mut_dptr<T>(), x->dptr<T>());
  } else if (like->shape().elem_cnt() == 1) {
    CHECK_EQ(n, x->shape().elem_cnt());
    KernelUtil<device_type, T>::Replicate(ctx.device_ctx, n, y->mut_dptr<T>(), like->dptr<T>());
  } else {
    size_t num_axes = y->shape().NumAxes();
    if (this->op_conf().broadcast_like_conf().has_kept_dims_shape()) {
      NdarrayUtil<device_type, T>::template BroadcastApply<BinaryFuncDiv>(
          ctx.device_ctx, XpuVarNdarray<T>(y, num_axes),
          XpuVarNdarray<const T>(Shape(this->op_conf().broadcast_like_conf().kept_dims_shape()),
                                 x->dptr<T>()),
          XpuVarNdarray<const T>(like, num_axes));
    } else {
      NdarrayUtil<device_type, T>::template BroadcastApply<BinaryFuncDiv>(
          ctx.device_ctx, XpuVarNdarray<T>(y, num_axes), XpuVarNdarray<const T>(x, num_axes),
          XpuVarNdarray<const T>(like, num_axes));
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastLikeConf, BroadcastLikeKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
