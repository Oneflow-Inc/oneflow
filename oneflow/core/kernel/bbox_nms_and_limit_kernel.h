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
  void BroadCastBboxTransform(const int64_t im_index,
                              const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void ClipBox(Blob* bbox_blob) const;
  void NmsAndTryVote(const int64_t im_index,
                     const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  int64_t Limit(const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void WriteOutputToOFRecord(int64_t image_index, int64_t limit_num,
                             const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void SortClassBoxIndexByScore(const T* scores_ptr, const int64_t boxes_num,
                                const int64_t class_num, const int64_t class_index,
                                int32_t* idx_ptr) const;
  int64_t FilterSortedIndexByThreshold(const int64_t num, const T* scores_ptr,
                                       const int32_t* idx_ptr, const float thresh) const;
  void BboxVoting(int64_t im_index, int64_t class_index, int32_t voter_num, int32_t votee_num,
                  const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  int64_t IndexMemContinuous(const int64_t class_num, const int64_t box_num,
                             const int32_t* post_nms_keep_num_ptr,
                             int32_t* post_nms_index_slice_ptr) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_KERNEL_H_
