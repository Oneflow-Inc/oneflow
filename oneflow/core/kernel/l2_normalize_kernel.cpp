#include "oneflow/core/kernel/l2_normalize_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf(), BnInOp2Blob("in"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf(), BnInOp2Blob("in"),
      BnInOp2Blob(GenDiffBn("out")), BnInOp2Blob(GenDiffBn("in")));
}

template<typename T>
struct L2NormalizeKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                      Blob* out_blob) {
    TODO();
  }

  static void Backward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    TODO();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL2NormalizeConf, L2NormalizeKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow