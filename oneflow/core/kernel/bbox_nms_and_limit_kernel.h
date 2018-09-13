#ifndef ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_
#define ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {

template<typename T>
class ScoringMethodIf {
 public:
  ScoringMethodIf() = default;
  virtual ~ScoringMethodIf() = default;
  void Init(const BboxVoteConf& vote_conf) { vote_conf_ = vote_conf; }
  const BboxVoteConf& conf() const { return vote_conf_; }
  virtual T scoring(
      const ScoredBoxesIndex<T>&, const T default_score,
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
  void ForwardDataId(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BroadcastBboxTransform(const int64_t im_index,
                              const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void ClipBox(Blob* bbox_blob) const;
  ScoredBoxesIndex<T> NmsAndTryVote(
      const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void VoteBboxAndScore(const ScoredBoxesIndex<T>& pre_nms_slice,
                        const ScoredBoxesIndex<T>& post_nms_slice, Blob* voting_score_blob,
                        Blob* bbox_blob) const;
  void VoteBbox(const ScoredBoxesIndex<T>& pre_nms_slice,
                const std::function<void(const std::function<void(int32_t, float)>&)>&,
                BBox<T>* ret_votee_bbox) const;
  void Limit(const int32_t limit_num, const float thresh, ScoredBoxesIndex<T>& slice) const;
  void WriteOutputToRecordBlob(const int64_t im_index, const int64_t boxes_num,
                               const ScoredBoxesIndex<T>& slice, Blob* labeled_bbox_blob,
                               Blob* bbox_score_blob) const;

  std::unique_ptr<ScoringMethodIf<T>> scoring_method_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_KERNEL_H_
