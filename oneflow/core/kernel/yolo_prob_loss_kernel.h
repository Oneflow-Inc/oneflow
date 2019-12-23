#ifndef ONEFLOW_CORE_KERNEL_YOLO_PROB_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_YOLO_PROB_LOSS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class YoloProbLossKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloProbLossKernel);
  YoloProbLossKernel() = default;
  ~YoloProbLossKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct YoloProbLossKernelUtil {
  static void CalcObjnessDiff(DeviceCtx* ctx, const size_t pos_num, const size_t neg_num,
                              const int32_t*, const int32_t* pos_inds_ptr,
                              const int32_t* neg_inds_ptr, const T* bbox_objness_ptr,
                              T* bbox_objness_out_ptr);
  static void CalcClsProbDiff(DeviceCtx* ctx, const size_t pos_num, const int32_t num_clsprobs,
                              const int32_t*, const int32_t* pos_inds_ptr,
                              const int32_t* pos_cls_label_ptr, const T* bbox_clsprob_ptr,
                              T* bbox_clsprob_out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_YOLO_PROB_LOSS_KERNEL_H_
