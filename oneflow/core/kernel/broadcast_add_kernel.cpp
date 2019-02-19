#include "oneflow/core/kernel/broadcast_add_kernel.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a_blob = BnInOp2Blob("a");
  const Blob* b_blob = BnInOp2Blob("b");
  Blob* out_blob = BnInOp2Blob("out");
  size_t num_axes = out_blob->shape().NumAxes();
  NdarrayUtil<device_type, T>::template BroadcastApply<BinaryFuncAdd>(
      kernel_ctx.device_ctx, XpuVarNdarray<T>(out_blob, num_axes),
      XpuVarNdarray<const T>(a_blob, num_axes), XpuVarNdarray<const T>(b_blob, num_axes));
}

template<DeviceType device_type, typename T>
void BroadcastAddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* a_diff_blob = BnInOp2Blob("a_diff");
  Blob* b_diff_blob = BnInOp2Blob("b_diff");
  Blob* bw_buf_blob = BnInOp2Blob("bw_buf");
  size_t num_axes = out_diff_blob->shape().NumAxes();
  if (a_diff_blob) {
    NdarrayUtil<device_type, T>::ReduceSum(
        kernel_ctx.device_ctx, XpuVarNdarray<T>(a_diff_blob, num_axes),
        XpuVarNdarray<const T>(out_diff_blob, num_axes), XpuVarNdarray<T>(bw_buf_blob, num_axes));
  }
  if (b_diff_blob) {
    NdarrayUtil<device_type, T>::ReduceSum(
        kernel_ctx.device_ctx, XpuVarNdarray<T>(b_diff_blob, num_axes),
        XpuVarNdarray<const T>(out_diff_blob, num_axes), XpuVarNdarray<T>(bw_buf_blob, num_axes));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastAddConf, BroadcastAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
