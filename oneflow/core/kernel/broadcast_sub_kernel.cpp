#include "oneflow/core/kernel/broadcast_sub_kernel.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/unary_func.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastSubKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a_blob = BnInOp2Blob("a");
  const Blob* b_blob = BnInOp2Blob("b");
  Blob* out_blob = BnInOp2Blob("out");
  size_t num_axes = out_blob->shape().NumAxes();
  NdarrayUtil<device_type, T>::BroadcastSub(
      kernel_ctx.device_ctx, XpuVarNdarray<T>(out_blob, num_axes),
      XpuVarNdarray<const T>(a_blob, num_axes), XpuVarNdarray<const T>(b_blob, num_axes));
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kBroadcastSubConf, BroadcastSubKernel);
}  // namespace oneflow
