#include "oneflow/core/kernel/roi_align_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RoIAlignKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* rois_blob = BnInOp2Blob("rois");
  Blob* out_blob = BnInOp2Blob("out");
  RoIAlignKernelUtil<device_type, T>::Forward(ctx, this->op_conf().roi_align_conf(), in_blob,
                                              rois_blob, out_blob);
}

template<DeviceType device_type, typename T>
void RoIAlignKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* rois_blob = BnInOp2Blob("rois");
  RoIAlignKernelUtil<device_type, T>::Backward(ctx, this->op_conf().roi_align_conf(), out_diff_blob,
                                               rois_blob, in_diff_blob);
}

template<DeviceType device_type, typename T>
void RoIAlignKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("rois"));
}

template<DeviceType device_type, typename T>
void RoIAlignKernel<device_type, T>::BackwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
struct RoIAlignKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob) {
    UNIMPLEMENTED();
  }
  static void Backward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* out_diff_blob,
                       const Blob* rois_blob, Blob* in_diff_blob) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRoiAlignConf, RoIAlignKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
