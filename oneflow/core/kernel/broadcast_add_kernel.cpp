#include "oneflow/core/kernel/broadcast_add_kernel.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/kernel/broadcast_add_xpu_util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BroadcastBinaryKernelUtil<device_type, T, BinaryFuncAdd>::Forward(kernel_ctx, BnInOp2Blob);
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
    CHECK_EQ(a_diff_blob->shape().NumAxes(), num_axes);
    SwitchBackwardInputDiff(
        SwitchCase(num_axes), kernel_ctx.device_ctx, XpuVarNdarray<T>(a_diff_blob, num_axes),
        XpuVarNdarray<const T>(out_diff_blob, num_axes), XpuVarNdarray<T>(bw_buf_blob, num_axes));
  }
  if (b_diff_blob) {
    CHECK_EQ(b_diff_blob->shape().NumAxes(), num_axes);
    SwitchBackwardInputDiff(
        SwitchCase(num_axes), kernel_ctx.device_ctx, XpuVarNdarray<T>(b_diff_blob, num_axes),
        XpuVarNdarray<const T>(out_diff_blob, num_axes), XpuVarNdarray<T>(bw_buf_blob, num_axes));
  }
}

template<typename T, int NDIMS>
struct BroadcastAddKernelUtil<DeviceType::kCPU, T, NDIMS> final {
  static void BackwardInputDiff(DeviceCtx*, XpuVarNdarray<T>&& in_diff_a,
                                const XpuVarNdarray<const T>& out_diff, XpuVarNdarray<T>&& bw_buf) {
    BroadcastAddXpuUtil<T, NDIMS>::BackwardInputDiff(&in_diff_a, out_diff, &bw_buf);
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastAddConf, BroadcastAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
