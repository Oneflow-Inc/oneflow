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

// todo: base class for roi pooling and roi align to generalize data id and col num
template<DeviceType device_type, typename T>
void RoIAlignKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* rois_blob = BnInOp2Blob("rois");
  Blob* out_blob = BnInOp2Blob("out");
  const int32_t roi_num = rois_blob->shape().At(1);
  const size_t size_of_one_data_id = Global<JobDesc>::Get()->SizeOfOneDataId();
  FOR_RANGE(int64_t, n, 0, in_blob->shape().At(0)) {
    const int32_t n_roi_offset = n * roi_num;
    FOR_RANGE(int64_t, r, 0, roi_num) {
      Memcpy<device_type>(ctx.device_ctx, out_blob->mut_data_id(n_roi_offset + r),
                          in_blob->data_id(n), size_of_one_data_id);
    }
  }
}

template<DeviceType device_type, typename T>
void RoIAlignKernel<device_type, T>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* rois_blob = BnInOp2Blob("rois");
  Blob* out_blob = BnInOp2Blob("out");
  const int32_t roi_num = rois_blob->shape().At(1);
  FOR_RANGE(int64_t, n, 0, in_blob->shape().At(0)) {
    const int32_t col_num = in_blob->col_num(n);
    const int32_t n_roi_offset = n * roi_num;
    FOR_RANGE(int64_t, r, 0, roi_num) { out_blob->set_col_num(n_roi_offset + r, col_num); }
  }
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
