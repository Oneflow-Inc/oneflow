#ifndef ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_
#define ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

template<typename T>
class ScoringMethodIf {
 public:
  ScoringMethodIf() = default;
  virtual ~ScoringMethodIf() = default;
  void Init(const BboxVoteConf& vote_conf) { vote_conf_ = vote_conf; }
  const BboxVoteConf& conf() const { return vote_conf_; }
  virtual T scoring(
      const T*, const int32_t votee_index,
      const std::function<void(const std::function<void(int32_t, float)>&)>&) const = 0;

 private:
  BboxVoteConf vote_conf_;
};

template<typename T>
class BboxNmsAndLimitKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BboxNmsAndLimitKernel);
  BboxNmsAndLimitKernel() = default;
  ~BboxNmsAndLimitKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext* parallel_ctx) override;
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
                  const int32_t* pre_nms_index_slice_ptr, const int32_t* post_nms_index_slice_ptr,
                  const int32_t* area_ptr, const Blob* score_blob, Blob* voting_score_blob,
                  Blob* bbox_blob) const;
  int64_t IndexMemContinuous(const int64_t class_num, const int64_t box_num,
                             const int32_t* post_nms_keep_num_ptr,
                             int32_t* post_nms_index_slice_ptr) const;

  std::unique_ptr<ScoringMethodIf<T>> scoring_method_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_KERNEL_H_
