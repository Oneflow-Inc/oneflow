#ifndef ONEFLOW_CORE_KERNEL_ROI_RESIZE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ROI_RESIZE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class RoIResizeKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIResizeKernel);
  RoIResizeKernel() = default;
  ~RoIResizeKernel() = default;

 private:
  void ForwardDataId(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
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

  void ForwardColNum(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
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
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ROI_RESIZE_KERNEL_H_
