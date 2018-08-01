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
  void SortClassBoxIndexByScore(const T* score_ptr, const int64_t box_num, const int64_t class_num,
                                const int64_t class_index, int32_t* pre_nms_index_slice) const;
  void BroadCastBboxTransform(const int64_t im_index,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void ClipBox(Blob* bbox_blob) const;
  void IndexMemContinuous(const int32_t* post_nms_keep_num_ptr, const int64_t class_num,
                          const int64_t box_num, int32_t* post_nms_index_slice_ptr) const;
  int64_t FilterSortedIndexByThreshold(const T* scores_ptr, const float score_thresh,
                                       const int32_t* pre_nms_index_slice, const int64_t num) const;
  void BboxVoting(int64_t class_index, int32_t voter_num,
                  std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void NmsAndTryVote(int64_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  int64_t Limit(std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void WriteOutputToOFRecord(int64_t image_index, int64_t limit_num,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_KERNEL_H_
