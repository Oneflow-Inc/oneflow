#include "oneflow/core/kernel/roi_resize_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void RoIResizeKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* rois_blob = BnInOp2Blob("rois");
  Blob* out_blob = BnInOp2Blob("out");
  const int32_t roi_num = rois_blob->shape().At(1);
  const size_t size_of_one_data_id = Global<JobDesc>::Get()->SizeOfOneDataId();
  FOR_RANGE(int64_t, n, 0, in_blob->shape().At(0)) {
    const int32_t n_roi_offset = n * roi_num;
    FOR_RANGE(int64_t, r, 0, roi_num) {
      memcpy(out_blob->mut_data_id(n_roi_offset + r), in_blob->data_id(n), size_of_one_data_id);
    }
  }
}

template<DeviceType device_type>
void RoIResizeKernel<device_type>::ForwardColNum(
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

template class RoIResizeKernel<DeviceType::kCPU>;
template class RoIResizeKernel<DeviceType::kGPU>;

}  // namespace oneflow
