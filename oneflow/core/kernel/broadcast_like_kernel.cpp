#include "oneflow/core/kernel/broadcast_like_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastLikeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x = BnInOp2Blob("x");
  Blob* y = BnInOp2Blob("y");
  const int64_t n = y->shape().elem_cnt();
  const int64_t num_axes = y->shape().NumAxes();
  if (this->op_conf().broadcast_like_conf().has_kept_dims_shape()) {
    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx.device_ctx, XpuVarNdarray<T>(y, num_axes),
        XpuVarNdarray<const T>(Shape(this->op_conf().broadcast_like_conf().kept_dims_shape()),
                               x->dptr<T>()));
  } else {
    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx.device_ctx, XpuVarNdarray<T>(y, num_axes),
        XpuVarNdarray<T>(x, num_axes);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastLikeConf, BroadcastLikeKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
