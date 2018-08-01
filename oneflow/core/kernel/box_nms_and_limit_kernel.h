#ifndef ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_
#define ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class BboxNmsAndLimitKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BboxNmsAndLimitKernel);
  BboxNmsAndLimitKernel() = default;
  ~BboxNmsAndLimitKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void SortClassBoxIndex(const T* score_ptr,const int32_t box_num, const int32_t class_num, const int32_t class_index, int32_t* pre_nms_index_slice);
  void BroadCastBboxTransform(const int32_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob);
  void ClipBox(Blob* bbox_blob);
  void FilterSortedIndexByThreshold(const T* scores_ptr,const T score_thresh,const int32_t* pre_nms_index_slice,const int32_t num);
  void NmsAndTryVote(int32_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob);
  int32_t Limit(std::function<Blob*(const std::string&)> BnInOp2Blob);
  void WriteOutputToOFRecord(int64_t image_index, in32_t limit_num, std::function<Blob*(const std::string&)> BnInOp2Blob)
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_KERNEL_H_
