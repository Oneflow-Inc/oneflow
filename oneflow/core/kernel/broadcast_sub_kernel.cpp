#include "oneflow/core/kernel/broadcast_sub_kernel.h"
#include "oneflow/core/kernel/broadcast_sub_xpu_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BroadcastSubKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BroadcastBinaryKernelUtil<device_type, T, BinaryFuncSub>::Forward(kernel_ctx, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void BroadcastSubKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* bw_buf_blob = BnInOp2Blob("bw_buf");
  Blob* a_diff_blob = BnInOp2Blob("a_diff");
  Blob* b_diff_blob = BnInOp2Blob("b_diff");
  size_t num_axes = out_diff_blob->shape().NumAxes();
  if (a_diff_blob) {
    SwitchBackwardInputDiffA(
        SwitchCase(num_axes), kernel_ctx.device_ctx, XpuVarNdarray<T>(a_diff_blob, num_axes),
        XpuVarNdarray<const T>(out_diff_blob, num_axes), XpuVarNdarray<T>(bw_buf_blob, num_axes));
  }
  if (b_diff_blob) {
    SwitchBackwardInputDiffB(
        SwitchCase(num_axes), kernel_ctx.device_ctx, XpuVarNdarray<T>(a_diff_blob, num_axes),
        XpuVarNdarray<const T>(out_diff_blob, num_axes), XpuVarNdarray<T>(bw_buf_blob, num_axes));
  }
}

template<typename T, int NDIMS>
struct BroadcastSubKernelUtil<DeviceType::kCPU, T, NDIMS> final {
  static void BackwardInputDiffA(DeviceCtx* ctx, XpuVarNdarray<T>&& in_diff,
                                 const XpuVarNdarray<const T>& out_diff,
                                 XpuVarNdarray<T>&& tmp_storage) {
    BroadcastSubXpuUtil<T, NDIMS>::BackwardInputDiffA(&in_diff, out_diff, &tmp_storage);
  }
  static void BackwardInputDiffB(DeviceCtx* ctx, XpuVarNdarray<T>&& in_diff,
                                 const XpuVarNdarray<const T>& out_diff,
                                 XpuVarNdarray<T>&& tmp_storage) {
    BroadcastSubXpuUtil<T, NDIMS>::BackwardInputDiffB(&in_diff, out_diff, &tmp_storage);
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastSubConf, BroadcastSubKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
